import torch as t
from torch.autograd import grad
from scipy.sparse.linalg import LinearOperator, eigsh
from train_mnist import CNN, load_mnist_data, load_pure_number_pattern_data, CombinedDataLoader, test
import argparse
import numpy as np
from helpers import orthogonal_complement, plot_pertubation_results
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


def get_hessian_eig_mnist(fname, patterns_per_num, opacity=0.5, use_mixed_dataloader=False):
    """
    Load model from fname checkpoint and calculate eigenvalues
    """
    model = CNN(input_size=28)
    model.load_state_dict(t.load(fname))
    model.to(device="cuda")
    model.eval()
    loss_fn = t.nn.CrossEntropyLoss()
    _, data_loader_test = load_mnist_data(patterns_per_num, opacity)
    if use_mixed_dataloader:
        d1, d2 = load_pure_number_pattern_data(patterns_per_num, is_train=False)
        data_loader_test = CombinedDataLoader(d1, d2)
    get_hessian_eigenvalues(model, loss_fn, data_loader_test, num_batches=30, device="cuda")


def eigenvector_similarity(fname, patterns_per_num, n_p=10):
    model = CNN(input_size=28)
    model.load_state_dict(t.load(fname))
    model.to(device="cuda")
    model.eval()
    loss_fn = t.nn.CrossEntropyLoss()
    # Load pure pattern number data
    data_loader_test_number, data_loader_test_pattern = load_pure_number_pattern_data(patterns_per_num, is_train=False)
    # get opacity 0.5 dataloader
    _, data_loader_05_test = load_mnist_data(patterns_per_num, opacity=0.5)

    _, _, eigenvectors_number = get_hessian_eigenvalues(model, loss_fn, data_loader_test_number, num_batches=50, device="cuda", n_top_vectors=n_p)
    _, _, eigenvectors_pattern = get_hessian_eigenvalues(model, loss_fn, data_loader_test_pattern, num_batches=50, device="cuda", n_top_vectors=n_p)
    _, _, eigenvectors_opacity_05 = get_hessian_eigenvalues(model, loss_fn, data_loader_05_test, num_batches=50, device="cuda", n_top_vectors=n_p)

    # get the dot products of all i,j eigenvectors for eigenvectors_number, eigenvectors_opacity_05 + save as image
    plt.clf()
    res = []
    for i in range(n_p):
        r = []
        for j in range(n_p):
            r.append(np.dot(eigenvectors_number[i], eigenvectors_opacity_05[j]))
        res.append(r)
    res = np.array(res)
    plt.imshow(res, cmap='gray')
    plt.savefig("eigenvector_similarity_number_opacity_05.png")

    # get the dot products of all i,j eigenvectors for eigenvectors_pattern, eigenvectors_opacity_05 + save as image
    plt.clf()
    res = []
    for i in range(n_p):
        r = []
        for j in range(n_p):
            r.append(np.dot(eigenvectors_pattern[i], eigenvectors_opacity_05[j]))
        res.append(r)
    res = np.array(res)
    plt.imshow(res, cmap='gray')
    plt.savefig("eigenvector_similarity_pattern_opacity_05.png")


def perturb_in_direction(fname, patterns_per_num, direction, n_p=50, just_return_proj_v=False):
    """
    fname: checkpoint file name
    patterns_per_num: number of patterns per digit
    direction: direction to perturb in ('pattern' or 'number')
    n_p: number of vectors to use for projection 
    """
    # Load model 
    model = CNN(input_size=28)
    model.load_state_dict(t.load(fname))
    model.to(device="cuda")
    model.eval()
    loss_fn = t.nn.CrossEntropyLoss()
    # Load pure pattern number data
    data_loader_test_number, data_loader_test_pattern = load_pure_number_pattern_data(patterns_per_num, is_train=False)

    _, _, eigenvectors_number = get_hessian_eigenvalues(model, loss_fn, data_loader_test_number, num_batches=50, device="cuda", n_top_vectors=n_p)
    _, _, eigenvectors_pattern = get_hessian_eigenvalues(model, loss_fn, data_loader_test_pattern, num_batches=50, device="cuda", n_top_vectors=n_p)

    # set eigenvectors_number and eigenvectors_pattern to random n_p x 3340 matrices
    # eigenvectors_number = np.random.rand(n_p, 3340)
    # eigenvectors_pattern = np.random.rand(n_p, 3340)

    if direction == 'number':
        orth = orthogonal_complement(eigenvectors_number) # 3340 x 3340
    elif direction == 'pattern':
        orth = orthogonal_complement(eigenvectors_pattern) # 3340 # 3340

    if direction == 'number':
        v = eigenvectors_pattern[-1] # 3340
    elif direction == 'pattern':
        v = eigenvectors_number[-1] # 3340
    
    proj_v = np.matmul(orth, v) # 3340

    if just_return_proj_v:
        return proj_v

    # get opacity 0.5 dataloader
    _, data_loader_05_test = load_mnist_data(patterns_per_num, opacity=0.5)


    t_values = np.linspace(0, 0.5, 50)

    # store results 
    acc_results = []
    loss_results = []
    for t_val in t_values:
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
        print(f"t_val: {t_val:.2f}, direction: {direction}, op_05_acc: {op_05_accuracy:.6f}, pure_num_acc: {pure_num_acc:.6f}, pure_pattern_acc: {pure_pattern_acc:.6f}, op_05_loss: {op_05_loss:.6f}, pure_num_loss: {pure_num_loss:.6f}, pure_pattern_loss: {pure_pattern_loss:.6f}")
        # store results
        acc_results.append((t_val, op_05_accuracy, pure_num_acc, pure_pattern_acc))
        loss_results.append((t_val, op_05_loss, pure_num_loss, pure_pattern_loss))

    # write results to textfile
    with open(f"txt_res/perturbation_results_{direction}_acc.txt", "w") as f:
        for result in acc_results:
            f.write(f"{result[0]},{result[1]},{result[2]},{result[3]}\n")
    with open(f"txt_res/perturbation_results_{direction}_loss.txt", "w") as f:
        for result in loss_results:
            f.write(f"{result[0]},{result[1]},{result[2]},{result[3]}\n")

    # plot results
    plot_pertubation_results(acc_results, 'perturbation_acc_results.png', yaxis='Accuracy (%)')
    plot_pertubation_results(loss_results, 'perturbation_loss_results.png', yaxis='Loss')





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

    # Save eigenvectors for future reference
    all_eigs_num = t.cat([t.tensor(e) for e in eigenvectors_number], dim=0)
    t.save(all_eigs_num, f"eigenvectors_number.pt")
    all_eigs_pat = t.cat([t.tensor(e) for e in eigenvectors_pattern], dim=0)
    t.save(all_eigs_pat, f"eigenvectors_pattern.pt")

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
    t_values = np.linspace(0, 0.5, 50)
    for idx, v in enumerate(v_dirs):
        proj_v = np.matmul(orth, v) # 3340
        # store results 
        acc_results = []
        loss_results = []
        for t_val in t_values:
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
            acc_results.append((t_val, op_05_accuracy, pure_num_acc, pure_pattern_acc))
            loss_results.append((t_val, op_05_loss, pure_num_loss, pure_pattern_loss))
        # write results to textfile
        with open(f"txt_res/perturbation_results_{direction}_acc_eig_{idx}.txt", "w") as f:
            for result in acc_results:
                f.write(f"{result[0]},{result[1]},{result[2]},{result[3]}\n")
        with open(f"txt_res/perturbation_results_{direction}_loss_eig_{idx}.txt", "w") as f:
            for result in loss_results:
                f.write(f"{result[0]},{result[1]},{result[2]},{result[3]}\n")
        # plot results
        plot_pertubation_results(acc_results, f'perturbation_acc_results_{direction}_eig_{idx}.png', yaxis='Accuracy (%)')
        plot_pertubation_results(loss_results, f'perturbation_loss_results_{direction}_eig_{idx}.png.png', yaxis='Loss')

def save_approx_hessian(fname, n=150, patterns_per_num=10):
    """
    Get top n eigenvectors for pure patterns, pure numbers, and mixed opacity 0.5 data
    Approximate hessian for each of these datasets using these eigenvectors
    See to what extent H_op_05 = H_num + H_pattern
    """
    # Load model
    model = CNN(input_size=28)
    model.load_state_dict(t.load(fname))
    model.to(device="cuda")
    model.eval()
    loss_fn = t.nn.CrossEntropyLoss()
    # Load pure pattern number data
    data_loader_test_number, data_loader_test_pattern = load_pure_number_pattern_data(patterns_per_num, is_train=False)
    # Load mixed opacity 0.5 data
    _, data_loader_05_test = load_mnist_data(patterns_per_num, opacity=0.5)
    # Get top n eigenvectors for each dataset
    _, eigenvalues_number, eigenvectors_number = get_hessian_eigenvalues(model, loss_fn, data_loader_test_number, num_batches=50, device="cuda", n_top_vectors=n)
    _, eigenvalues_pattern, eigenvectors_pattern = get_hessian_eigenvalues(model, loss_fn, data_loader_test_pattern, num_batches=50, device="cuda", n_top_vectors=n)
    _, eigenvalues_05, eigenvectors_05 = get_hessian_eigenvalues(model, loss_fn, data_loader_05_test, num_batches=50, device="cuda", n_top_vectors=n)
    
    # Function to get hessian given eigenvectors and eigenvalues
    def get_hessian(eigenvectors, eigenvalues):
        # Ensure inputs are numpy arrays
        eigenvectors = np.array(eigenvectors)
        eigenvalues = np.array(eigenvalues)

        # Check if eigenvectors and eigenvalues have correct sizes
        assert len(eigenvectors) == len(eigenvalues), "The number of eigenvectors and eigenvalues must match"

        # Create diagonal matrix from eigenvalues
        diag_matrix = np.diag(eigenvalues)

        # Reconstruct the matrix using spectral theorem (A = QΛQ')
        approx_matrix = eigenvectors.T @ diag_matrix @ eigenvectors

        return approx_matrix
    # Get Hessian for each dataset
    H_number = get_hessian(eigenvectors_number, eigenvalues_number)
    H_pattern = get_hessian(eigenvectors_pattern, eigenvalues_pattern)
    H_05 = get_hessian(eigenvectors_05, eigenvalues_05)
    # Create file to store results
    with open(f"txt_res/hessian_matrices_results_10patterns.txt", "w") as f:
        f.write(f"H_number:\n{H_number}\n\nH_pattern:\n{H_pattern}\n\nH_05:\n{H_05}\n")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("version", help="version of model to load")
#     parser.add_argument("--preserve", help="preserve number or pattern performance", type=str, default=None, required=False)
#     parser.add_argument("--opacity", help="opacity of patterns in loss data", type=float, default=0.5, required=False)
#     parser.add_argument("--patterns_per_num", help="number of patterns per digit", type=int, default=10, required=False)
#     parser.add_argument("--mixed", help="use mixed data loader", action="store_true", default=False, required=False)
#     args = parser.parse_args()
#     version = args.version
#     opacity = args.opacity
#     patterns_per_num = args.patterns_per_num
#     use_mixed_dataloader = args.mixed
#     if args.preserve is not None:
#         perturb_in_direction(f"./models/model_{version}.ckpt", patterns_per_num, args.preserve)
#     else:
#         get_hessian_eig_mnist(f"./models/model_{version}.ckpt", patterns_per_num=patterns_per_num, opacity=opacity, use_mixed_dataloader=use_mixed_dataloader)


if __name__ == "__main__":
    perturb_in_direction_per_eig("models/model_final_finetuned.ckpt", 10, "pattern", n_eig_dirs=10, n_orth_dirs=50)