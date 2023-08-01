import torch as t
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from train_mnist import CNN, load_mnist_data, test_pure_and_opacity, load_pure_number_pattern_data
from hessian_eig import get_hessian_eigenvalues
from tqdm import tqdm
from helpers import reshape_submodule_param_vector

def get_module_parameters(model):
    return model.conv2.parameters()

def loss_diff(model, input_data, target_data, difference_vector, loss_fn):
    original_loss = loss_fn(model(input_data), target_data)
    params_vector = parameters_to_vector(get_module_parameters(model)).detach()
    perturbed_params_vector = params_vector + difference_vector
    #[p + d for p, d in zip(params_vector(), difference_vector)]
    perturbed_params = vector_to_parameters(perturbed_params_vector)
    perturbed_loss = loss_fn(model(input_data, params=perturbed_params), target_data)
    return original_loss - perturbed_loss

def sphere_localized_loss_adjustment(model, top_eigenvectors, offset, radius = 1, lambda_sphere = 10, lambda_orth = .1, device='cuda'):
    """
    model: cnn
    top_eigenvectors: top eigenvectors of the Hessian on the mixed data problem [n_eigenvectors x n_params] (arranged in small-to-large order)
    offset: weight vector where we start
    radius: radius of the sphere we search on
    lambda_sphere: how much to penalize being outside the sphere
    lambda_orth: how much to penalize being inside the orthogonal complement of the top eigenvectors
    """
    top_eigenvectors = t.tensor(top_eigenvectors).to(device)
    offset = offset.to(device)
    proj_matrix = t.mm(top_eigenvectors.T, top_eigenvectors) # (n_params x n_eigenvectors) @ (n_eigenvectors x n_params) -> (n_params x n_params)
    params_vector = parameters_to_vector(model.parameters())
    params_proj = t.mv(proj_matrix, params_vector)
    offset_proj = t.mv(proj_matrix, offset)
    r_proj_params = t.sqrt(t.norm(params_proj-offset_proj))
    sphere_reg = lambda_sphere*(r_proj_params-radius)**2
    orth_reg = lambda_orth*t.norm(params_vector-offset-params_proj+offset_proj)
    return sphere_reg, orth_reg



def train_in_sphere(model, dataloader, top_eigenvectors, radius = 1, lambda_sphere = 10, lambda_orth = .1, lr = 1e-3, n_epochs = 3, device = 'cuda', patterns_per_num = 10, lr_decay=0.8):
    model.to(device)
    offset = parameters_to_vector(model.parameters()).detach()

    optimizer = t.optim.SGD(get_module_parameters(model), lr=lr)

    # reshape all eigenvectors to be the same shape as the model parameters
    top_eigenvectors = t.stack([reshape_submodule_param_vector(model, get_module_parameters, v) for v in top_eigenvectors])

    scheduler = t.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    loss_fn = t.nn.CrossEntropyLoss()

    # Adjust model weights to be on the sphere of high eigenvectors
    n_eigenvectors = top_eigenvectors.shape[0]
    rand_vec = t.randn(n_eigenvectors)
    unit_sphere_vec = rand_vec @ top_eigenvectors 
    unit_sphere_vec /= t.norm(unit_sphere_vec)
    point_on_sphere = offset.to(device) + radius*unit_sphere_vec.to(device)

    # v = t.load('proj_v_pattern.pt')
    # v = (v/t.norm(v, p=2)).float()
    # point_on_sphere = offset.to(device) + radius*v.to(device)

    # load the point on the sphere into the model
    vector_to_parameters(point_on_sphere, model.parameters())
    model.train()

    for epoch in range(n_epochs):
        idx = 0
        tot_sphere_reg = 0
        tot_orth_reg = 0
        tot_ce_loss = 0
        for batch, labels in tqdm(dataloader):
            idx += 1
            optimizer.zero_grad()

            sphere_reg, orth_reg = sphere_localized_loss_adjustment(model, top_eigenvectors, offset, radius, lambda_sphere, lambda_orth, device=device)
            loss_main = loss_fn(model(batch.to(device)), labels.to(device))
            
            tot_sphere_reg += sphere_reg.item()
            tot_orth_reg += orth_reg.item()
            tot_ce_loss += loss_main.item()

            loss = loss_main + sphere_reg + orth_reg

            if idx % 100 == 0:
                print(f"Epoch {epoch+1}, batch {idx}: avg_sphere_reg = {tot_sphere_reg/100}, avg_orth_reg = {tot_orth_reg/100}, avg_ce_loss = {tot_ce_loss/100}")
                tot_sphere_reg = 0
                tot_orth_reg = 0
                tot_ce_loss = 0
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} complete")
        test_pure_and_opacity(model, patterns_per_num, device=device)
        scheduler.step()
     
    # Save model as checkpoint
    t.save(model.state_dict(), 'sphere_model_2.pth')
    return model

def main():
    # Load CNN from good_models/model_final_finetuned.ckpt
    model = CNN(input_size=28)
    model.load_state_dict(t.load("good_models/model_final_finetuned.ckpt"))
    model.to(device="cuda")
    model.eval()
    loss_fn = t.nn.CrossEntropyLoss()
    # Get opacity 0.5 data
    data_loader_train, data_loader_test = load_mnist_data(patterns_per_num=10, opacity=0.5)
    _, _, eigenvectors = get_hessian_eigenvalues(model, loss_fn, data_loader_test, device="cuda", n_top_vectors=100, param_extract_fn=get_module_parameters)
    # Train in sphere
    train_in_sphere(model, data_loader_train, eigenvectors, radius=2, lambda_sphere=10, lambda_orth=1, lr=0.01, n_epochs=10, device="cuda", patterns_per_num=10)

def main2():
    # Load CNN from good_models/model_final_finetuned.ckpt
    model = CNN(input_size=28)
    model.load_state_dict(t.load("models/model_final_finetuned_avgpool.ckpt"))
    model.to(device="cuda")
    model.eval()
    loss_fn = t.nn.CrossEntropyLoss()
    # Get pure number data
    data_loader_test_number, data_loader_test_pattern = load_pure_number_pattern_data(patterns_per_num=10, is_train=False)
    _, _, eigenvectors = get_hessian_eigenvalues(model, loss_fn, data_loader_test_number, device="cuda", n_top_vectors=5, param_extract_fn=get_module_parameters)
    # Get opacity 0.5 data
    data_loader_train, _ = load_mnist_data(patterns_per_num=10, opacity=0.5)
    # Train in sphere
    train_in_sphere(model, data_loader_train, eigenvectors, radius=.5, lambda_sphere=10, lambda_orth=2, lr=0.01, n_epochs=10, device="cuda", patterns_per_num=10)


if __name__ == '__main__':
    main2()