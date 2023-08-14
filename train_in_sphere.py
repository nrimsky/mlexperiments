import torch as t
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from train_mnist import CNN, load_mnist_data, test_pure_and_opacity, load_pure_number_pattern_data, test
from hessian_eig import get_hessian_eigenvalues
from tqdm import tqdm
from helpers import reshape_submodule_param_vector, plot_semi_supervised_results, plot_unsupervised_results
import numpy as np

def get_module_parameters(model):
    return model.parameters()

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
    r_proj_params = t.norm(params_proj-offset_proj)
    sphere_reg = lambda_sphere*(r_proj_params-radius)**2
    orth_reg = lambda_orth*(t.norm(params_vector-offset-params_proj+offset_proj)**2)
    return sphere_reg, orth_reg


def project_to_subspace(parameters, eigenvectors, eigenvalues):
    """
    Project the parameters onto the subspace spanned by the eigenvectors.
    Weigh the projection by the importance of each eigendirection (eigenvalue).
    """
    projections = [eigenvalues[i] * t.dot(parameters, vec) * vec for i, vec in enumerate(eigenvectors)]
    return sum(projections)

def test_eigen_model(model, dataset1, dataset2, n_top_vectors=5, n_batches=10, device="cuda"):
    """
    model: model trained on mixed data
    dataset1: dataset to get internal activations and eigenvectors on
    dataset2: dataset to get internal activations on for comparison
    n_top_vectors: number of eigenvectors to use for reconstruction
    n_batches: number of batches to use for activation comparison

    1. Get internal activations of model on subset of dataset1 and dataset2
    2. Get n_top_vectors top eigenvectors of the Hessian of the model loss on the dataset1
    3. Project model weights to the the n_top_vectors top eigenvectors
    4. Get internal activations of reconstructed model on subset of dataset1 and dataset2
    5. Check similarity between internal activations on dataset1 before reconstruct vs dataset1 after reconstruct, and dataset2 before reconstruct vs dataset2 after reconstruct
    """

    loss_fn = t.nn.CrossEntropyLoss()

    subset_images_1, subset_images_2 = [], []
    subset_targets_1, subset_targets_2 = [], []
    idx = 0 
    for (images1, targets1), (images2, targets2) in zip(dataset1, dataset2):
        idx += 1
        if idx > n_batches:
            break
        subset_images_1.append(images1.to(device))
        subset_images_2.append(images2.to(device))
        subset_targets_1.append(targets1.to(device))
        subset_targets_2.append(targets2.to(device))
    subset_images_1 = t.cat(subset_images_1)
    subset_images_2 = t.cat(subset_images_2)
    subset_targets_1 = t.cat(subset_targets_1)
    subset_targets_2 = t.cat(subset_targets_2)

    out1 = model(subset_images_1)
    loss1 = loss_fn(out1, subset_targets_1)
    activations1 = model.get_all_internal_states()

    out2 = model(subset_images_2)
    loss2 = loss_fn(out2, subset_targets_2)
    activations2 = model.get_all_internal_states()


    _, eigenvalues, eigenvectors = get_hessian_eigenvalues(model, t.nn.CrossEntropyLoss(), dataset1, device=device, n_top_vectors=n_top_vectors, param_extract_fn=get_module_parameters, num_batches=n_batches)
    eigenvectors = t.tensor(eigenvectors).to(device)

    # Flatten model parameters
    flat_parameters = parameters_to_vector(model.parameters())

    # Project the model's parameters onto the subspace spanned by the top eigenvectors
    params_reconstruct = project_to_subspace(flat_parameters, eigenvectors, eigenvalues)

    # Normalize params_reconstruct to have same norm as original parameters
    params_reconstruct = (params_reconstruct * t.norm(flat_parameters)) / t.norm(params_reconstruct)

    vector_to_parameters(params_reconstruct, model.parameters())

    out1_r = model(subset_images_1)
    loss1_r = loss_fn(out1_r, subset_targets_1)
    activations1_reconstruct = model.get_all_internal_states()

    out2_r = model(subset_images_2)
    loss2_r = loss_fn(out2_r, subset_targets_2)
    activations2_reconstruct = model.get_all_internal_states()

    # Check similarity between activations1 and activations1_reconstruct
    # Expect this to be higher as we are using the eigenvectors on dataset1
    similarity = t.cosine_similarity(t.tensor(activations1), t.tensor(activations1_reconstruct), dim=0)
    print("Similarity between activations1 and activations1_reconstruct: ", similarity)

    # Check similarity between activations2 and activations2_reconstruct
    similarity = t.cosine_similarity(t.tensor(activations2), t.tensor(activations2_reconstruct), dim=0)
    print("Similarity between activations2 and activations2_reconstruct: ", similarity)

    # Print losses
    print("Loss on dataset1 before reconstruct: ", loss1.item())
    print("Loss on dataset1 after reconstruct: ", loss1_r.item())
    print("Loss on dataset2 before reconstruct: ", loss2.item())
    print("Loss on dataset2 after reconstruct: ", loss2_r.item())



def train_in_sphere(model, dataloader, top_eigenvectors, radius = 1, lambda_sphere = 10, lambda_orth = .1, lr = 1e-3, n_epochs = 3, device = 'cuda', patterns_per_num = 10, lr_decay=0.8):
    model.to(device)
    offset = parameters_to_vector(model.parameters()).detach()

    optimizer = t.optim.SGD(get_module_parameters(model), lr=lr, weight_decay=0.0)

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

def unsupervised():
    model = CNN(input_size=28)
    model.load_state_dict(t.load("models/model_final_finetuned.ckpt"))
    model.to(device="cuda")
    model.eval()
    loss_fn = t.nn.CrossEntropyLoss()
    radii = [0.5, 1, 1.5, 2, 2.5, 3]
    data_loader_test_number, data_loader_test_pattern = load_pure_number_pattern_data(patterns_per_num=10, is_train=False)
    data_loader_train, data_loader_test = load_mnist_data(patterns_per_num=10, opacity=0.5)
    _, _, eigenvectors = get_hessian_eigenvalues(model, loss_fn, data_loader_train, device="cuda", n_top_vectors=35, param_extract_fn=get_module_parameters)
    results = []
    with open('results_unsupervised.txt', 'w') as f:
        for radius in radii:
            model = CNN(input_size=28)
            model.load_state_dict(t.load("models/model_final_finetuned.ckpt"))
            model.to(device="cuda")
            model.eval()
            train_in_sphere(model, data_loader_train, eigenvectors, radius=radius, lambda_sphere=15, lambda_orth=1, lr=0.005, n_epochs=10, device="cuda", patterns_per_num=10)
            sphere_model = CNN(input_size=28)
            sphere_model.load_state_dict(t.load("sphere_model_2.pth"))
            sphere_model.to(device="cuda")
            sphere_model.eval()
            acc_number, loss_number = test(sphere_model, data_loader_test_number, device="cuda", calc_loss=True, do_print=False)
            acc_pattern, loss_pattern = test(sphere_model, data_loader_test_pattern, device="cuda", calc_loss=True, do_print=False)
            acc_op, loss_op = test(sphere_model, data_loader_test, device="cuda", calc_loss=True, do_print=False)
            print(f"Radius: {radius}, acc_number: {acc_number}, loss_number: {loss_number}, acc_pattern: {acc_pattern}, loss_pattern: {loss_pattern}, acc_op: {acc_op}, loss_op: {loss_op}")
            results.append((radius, acc_number, loss_number, acc_pattern, loss_pattern, acc_op, loss_op))
            f.write(f"Radius: {radius}, acc_number: {acc_number}, loss_number: {loss_number}, acc_pattern: {acc_pattern}, loss_pattern: {loss_pattern}, acc_op: {acc_op}, loss_op: {loss_op}\n")
    plot_unsupervised_results(results)


def semi_supervised(sphere):
    """
    sphere: 'number' or 'pattern' - which eigenvectors to use
    TODO: reduce radius range for `number` sphere as it's more sensitive to radius -> contributes lower eigenvalues??
    """
    model = CNN(input_size=28)
    model.load_state_dict(t.load("models/model_final_finetuned.ckpt"))
    model.to(device="cuda")
    model.eval()
    loss_fn = t.nn.CrossEntropyLoss()
    radii = [0.5, 1, 1.5, 2, 2.5, 3]
    eigen_loader = None
    data_loader_test_number, data_loader_test_pattern = load_pure_number_pattern_data(patterns_per_num=10, is_train=False)
    if sphere == 'number':
        eigen_loader = data_loader_test_number
    elif sphere == 'pattern':
        eigen_loader = data_loader_test_pattern
    else:
        raise ValueError("sphere must be 'number' or 'pattern'")
    _, _, eigenvectors = get_hessian_eigenvalues(model, loss_fn, eigen_loader, device="cuda", n_top_vectors=35, param_extract_fn=get_module_parameters)
    data_loader_train, data_loader_test = load_mnist_data(patterns_per_num=10, opacity=0.5)
    results = []
    with open(f'results_{sphere}.txt', 'w') as f:
        for radius in radii:
            model = CNN(input_size=28)
            model.load_state_dict(t.load("models/model_final_finetuned.ckpt"))
            model.to(device="cuda")
            model.eval()
            train_in_sphere(model, data_loader_train, eigenvectors, radius=radius, lambda_sphere=15, lambda_orth=1, lr=0.005, n_epochs=10, device="cuda", patterns_per_num=10)
            sphere_model = CNN(input_size=28)
            sphere_model.load_state_dict(t.load("sphere_model_2.pth"))
            sphere_model.to(device="cuda")
            sphere_model.eval()
            acc_number, loss_number = test(sphere_model, data_loader_test_number, device="cuda", calc_loss=True, do_print=False)
            acc_pattern, loss_pattern = test(sphere_model, data_loader_test_pattern, device="cuda", calc_loss=True, do_print=False)
            acc_op, loss_op = test(sphere_model, data_loader_test, device="cuda", calc_loss=True, do_print=False)
            print(f"Radius: {radius}, acc_number: {acc_number}, loss_number: {loss_number}, acc_pattern: {acc_pattern}, loss_pattern: {loss_pattern}, acc_op: {acc_op}, loss_op: {loss_op}")
            results.append((radius, acc_number, loss_number, acc_pattern, loss_pattern, acc_op, loss_op))
            f.write(f"Radius: {radius}, acc_number: {acc_number}, loss_number: {loss_number}, acc_pattern: {acc_pattern}, loss_pattern: {loss_pattern}, acc_op: {acc_op}, loss_op: {loss_op}\n")
    plot_semi_supervised_results(results, sphere)

def activation_similarity_eigenmodel():
    model = CNN(input_size=28)
    model.load_state_dict(t.load("models/model_final_finetuned.ckpt"))
    model.to(device="cuda")
    model.eval()
    data_loader_test_number, data_loader_test_pattern = load_pure_number_pattern_data(patterns_per_num=10, is_train=False)
    test_eigen_model(model, dataset1=data_loader_test_pattern, dataset2=data_loader_test_number, n_top_vectors=1, n_batches=20, device="cuda")

if __name__ == '__main__':
    # semi_supervised('pattern')
    semi_supervised('number')
    unsupervised()