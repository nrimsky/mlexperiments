import torch as t
from torch.autograd import grad
from scipy.sparse.linalg import LinearOperator, eigsh
from train_mnist import CNN, load_mnist_data, load_pure_number_pattern_data, test
import numpy as np
from helpers import orthogonal_complement
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from matplotlib import pyplot as plt

def get_hessian_eigenvalues(model, loss_fn, train_data_loader, num_batches=30, device="cuda", n_top_vectors=200, param_extract_fn=None):
    """
    model: a pytorch model
    loss_fn: a pytorch loss function
    train_data_loader: a pytorch data loader
    num_batches: number of batches to use for the hessian calculation
    device: the device to use for the hessian calculation
    """
    param_extract_fn = param_extract_fn or (lambda x: x.parameters())
    num_params = sum(p.numel() for p in param_extract_fn(model))
    subset_images, subset_labels = [], []
    for batch_idx, (images, labels) in enumerate(train_data_loader):
        if batch_idx >= num_batches:
            break
        subset_images.append(images.to(device))
        subset_labels.append(labels.to(device))
    subset_images = t.cat(subset_images)
    subset_labels = t.cat(subset_labels)

    def compute_loss():
        output = model(subset_images)
        return loss_fn(output, subset_labels)
    
    def hessian_vector_product(vector):
        model.zero_grad()
        grad_params = grad(compute_loss(), param_extract_fn(model), create_graph=True)
        flat_grad = t.cat([g.view(-1) for g in grad_params])
        grad_vector_product = t.sum(flat_grad * vector)
        hvp = grad(grad_vector_product, param_extract_fn(model), retain_graph=True)
        return t.cat([g.contiguous().view(-1) for g in hvp])
    
    def matvec(v):
        v_tensor = t.tensor(v, dtype=t.float32, device=device)
        return hessian_vector_product(v_tensor).cpu().detach().numpy()
    
    linear_operator = LinearOperator((num_params, num_params), matvec=matvec)
    eigenvalues, eigenvectors = eigsh(linear_operator, k=n_top_vectors, tol=0.001, which='LM', return_eigenvectors=True)
    tot = 0
    thresholds = [0.1, 1, 2, 10]
    for e in eigenvalues:
        print("{:.2f}".format(e))
    for threshold in thresholds:    
        for e in eigenvalues:
            if e > threshold:
                tot += 1
        print(f"Number of eigenvalues greater than {threshold}: {tot}")
        tot = 0
    eigenvectors = np.transpose(eigenvectors)
    return tot, eigenvalues, eigenvectors



def perturb_in_direction_per_eig(fname, patterns_per_num, direction, n_eig_dirs=50, n_orth_dirs=50):
    """
    fname: checkpoint file name
    patterns_per_num: number of patterns per digit
    direction: direction to perturb in ('pattern' or 'number')
    n_eig_dirs: number of vectors to use for projection to high eigenvector manifold
    n_orth_dirs: number of vectors to use for producing orthogonal complement as generalization direction
    """
    # Load model 
    model = CNN(input_size=28)
    model.load_state_dict(t.load(fname))
    model.to(device="cuda")
    model.eval()
    loss_fn = t.nn.CrossEntropyLoss()
    # Load pure pattern number data
    data_loader_test_number, data_loader_test_pattern = load_pure_number_pattern_data(patterns_per_num, is_train=False)

    if direction == 'number':
        _, _, eigenvectors_number = get_hessian_eigenvalues(model, loss_fn, data_loader_test_number, num_batches=50, device="cuda", n_top_vectors=n_orth_dirs)
        _, _, eigenvectors_pattern = get_hessian_eigenvalues(model, loss_fn, data_loader_test_pattern, num_batches=50, device="cuda", n_top_vectors=n_eig_dirs)
    elif direction == 'pattern':
        _, _, eigenvectors_number = get_hessian_eigenvalues(model, loss_fn, data_loader_test_number, num_batches=50, device="cuda", n_top_vectors=n_eig_dirs)
        _, _, eigenvectors_pattern = get_hessian_eigenvalues(model, loss_fn, data_loader_test_pattern, num_batches=50, device="cuda", n_top_vectors=n_orth_dirs)

    if direction == 'number':
        orth = orthogonal_complement(eigenvectors_number) # 3340 x 3340
    elif direction == 'pattern':
        orth = orthogonal_complement(eigenvectors_pattern) # 3340 # 3340

    
    if direction == 'number':
        v_dirs = eigenvectors_pattern[-n_eig_dirs:] # 3340
    elif direction == 'pattern':
        v_dirs = eigenvectors_number[-n_eig_dirs:] # 3340
    
    # get opacity 0.5 dataloader
    _, data_loader_05_test = load_mnist_data(patterns_per_num, opacity=0.5)
    t_val = 1.0
    acc_results = []
    loss_results = []
    for idx, v in enumerate(v_dirs[::-1]):
        proj_v = np.matmul(orth, v) # 3340
        # load model 
        model = CNN(input_size=28)
        model.load_state_dict(t.load(fname))
        model.eval()
        model.to(device="cuda")
        # perturb parameters by t * proj_v
        params_vector = parameters_to_vector(model.parameters()).detach()
        pertubation = t.tensor(t_val * proj_v, dtype=t.float32).to(device="cuda")
        params_vector = params_vector + pertubation
        vector_to_parameters(params_vector, model.parameters())
        # evaluating model 
        op_05_accuracy, op_05_loss = test(model, data_loader_05_test, do_print=False, device="cuda", calc_loss=True, max_batches=100)
        pure_num_acc, pure_num_loss = test(model, data_loader_test_number, do_print=False, device="cuda", calc_loss=True, max_batches=100)
        pure_pattern_acc, pure_pattern_loss = test(model, data_loader_test_pattern, do_print=False, device="cuda", calc_loss=True, max_batches=100)
        # print results
        print(f"EIG {idx} | t_val: {t_val:.2f}, direction: {direction}, op_05_acc: {op_05_accuracy:.6f}, pure_num_acc: {pure_num_acc:.6f}, pure_pattern_acc: {pure_pattern_acc:.6f}, op_05_loss: {op_05_loss:.6f}, pure_num_loss: {pure_num_loss:.6f}, pure_pattern_loss: {pure_pattern_loss:.6f}")
        # store results
        acc_results.append((idx, t_val, op_05_accuracy, pure_num_acc, pure_pattern_acc))
        loss_results.append((idx, t_val, op_05_loss, pure_num_loss, pure_pattern_loss))
        # Extracting values from results for plotting
    
    eig_indices = [result[0] for result in acc_results]
    op_05_accuracies = [result[2] for result in acc_results]
    pure_num_accuracies = [result[3] for result in acc_results]
    pure_pattern_accuracies = [result[4] for result in acc_results]

    op_05_losses = [result[2] for result in loss_results]
    pure_num_losses = [result[3] for result in loss_results]
    pure_pattern_losses = [result[4] for result in loss_results]

    # Plotting accuracies
    plt.clf()
    plt.figure()
    plt.plot(eig_indices, op_05_accuracies, 'o', label='Opacity 0.5 Accuracy')
    plt.plot(eig_indices, pure_num_accuracies, 'o', label='Number Accuracy', color='red')
    plt.plot(eig_indices, pure_pattern_accuracies, 'o', label='Pattern Accuracy', color='green')
    plt.xlabel('Eigenvector Index')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Loss vs Eig Idx (steering to preserve {direction}, t={t_val:.2f})')
    plt.grid(True)
    plt.savefig(f"eigenvector_index_vs_accuracy_{direction}_{t_val}.png")

    # Plotting losses
    plt.clf()
    plt.figure()
    plt.plot(eig_indices, op_05_losses, 'o', label='Opacity 0.5 Loss')
    plt.plot(eig_indices, pure_num_losses, 'o', label='Number Loss', color='red')
    plt.plot(eig_indices, pure_pattern_losses, 'o', label='Pattern Loss', color='green')
    plt.xlabel('Eigenvector Index')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Loss vs Eig Idx (steering to preserve {direction}, t={t_val:.2f})')
    plt.grid(True)
    plt.savefig(f"eigenvector_index_vs_loss_{direction}_{t_val}.png")


if __name__ == "__main__":
    perturb_in_direction_per_eig("models/model_final_finetuned.ckpt", 10, "pattern", n_eig_dirs=50, n_orth_dirs=50)