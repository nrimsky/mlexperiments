import torch as t
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from hessian_eig import get_hessian_eigenvalues
from tqdm import tqdm
from helpers import reshape_submodule_param_vector
from mlp_modular import MLP, test_model, get_train_test_loaders

def get_module_parameters(model):
    return model.fc.parameters()

def sphere_localized_loss_adjustment(model, top_eigenvectors, offset, radius = 1, lambda_sphere = 10, lambda_orth = .1, device='cuda'):
    """
    model: mlp
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

def train_in_sphere(model, dataloader, top_eigenvectors, radius = 1, lambda_sphere = 10, lambda_orth = .1, lr = 1e-3, n_epochs = 3, device = 'cuda', lr_decay=0.8):
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
        test_model(model, dataloader, device=device, criterion=loss_fn)
        scheduler.step()
     
    # Save model as checkpoint
    t.save(model.state_dict(), 'modular_addition_sphere_model.pth')
    return model

def main():
    model = MLP(vocab_size=114, embed_dim=14, hidden_dim=8)
    model.load_state_dict(t.load("modular_addition.ckpt"))
    model.to(device="cuda")
    model.eval()
    loss_fn = t.nn.CrossEntropyLoss()
    train_loader, test_loader = get_train_test_loaders(train_frac=0.4, batch_size=256, vocab_size=114)
    _, _, eigenvectors = get_hessian_eigenvalues(model, loss_fn, test_loader, device="cuda", n_top_vectors=5, param_extract_fn=get_module_parameters)
    train_in_sphere(model, train_loader, eigenvectors, radius=.5, lambda_sphere=10, lambda_orth=2, lr=0.01, n_epochs=10, device="cuda")


if __name__ == '__main__':
    main()