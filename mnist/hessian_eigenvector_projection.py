import torch as t
from train_mnist import CNN, load_mnist_data, load_pure_number_pattern_data, CombinedDataLoader, test
import argparse
import numpy as np
from utils import orthogonal_complement, plot_pertubation_results, plot_acc_perturb_in_direction_per_eig, plot_loss_perturb_in_direction_per_eig
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from matplotlib import pyplot as plt
from hessian_eig import hessian_eig
from visualize_conv_kernels import make_conv_movies, save_frames_weights


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
        _, _, eigenvectors_number = hessian_eig(model, loss_fn, data_loader_test_number, num_batches=50, device="cuda", n_top_vectors=n_orth_dirs)
        _, _, eigenvectors_pattern = hessian_eig(model, loss_fn, data_loader_test_pattern, num_batches=50, device="cuda", n_top_vectors=n_eig_dirs)
    elif direction == 'pattern':
        _, _, eigenvectors_number = hessian_eig(model, loss_fn, data_loader_test_number, num_batches=50, device="cuda", n_top_vectors=n_eig_dirs)
        _, _, eigenvectors_pattern = hessian_eig(model, loss_fn, data_loader_test_pattern, num_batches=50, device="cuda", n_top_vectors=n_orth_dirs)

    if direction == 'number':
        orth = orthogonal_complement(eigenvectors_number) # 3340 x 3340
    elif direction == 'pattern':
        orth = orthogonal_complement(eigenvectors_pattern) # 3340 # 3340

    
    if direction == 'number':
        v_dirs = eigenvectors_pattern[-n_eig_dirs:] # 3340
    elif direction == 'pattern':
        v_dirs = eigenvectors_number[-n_eig_dirs:] # 3340

    # Save eigenvectors for future reference
    all_eigs_num = t.cat([t.tensor(e) for e in eigenvectors_number], dim=0)
    t.save(all_eigs_num, f"eigenvectors_number.pt")
    all_eigs_pat = t.cat([t.tensor(e) for e in eigenvectors_pattern], dim=0)
    t.save(all_eigs_pat, f"eigenvectors_pattern.pt")
    
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
    plot_acc_perturb_in_direction_per_eig(eig_indices, op_05_accuracies, pure_num_accuracies, pure_pattern_accuracies, direction, t_val)

    # Plotting losses
    plot_loss_perturb_in_direction_per_eig(eig_indices, op_05_losses, pure_num_losses, pure_pattern_losses, direction, t_val)


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
    hessian_eig(model, loss_fn, data_loader_test, num_batches=30, device="cuda")


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

    _, _, eigenvectors_number = hessian_eig(model, loss_fn, data_loader_test_number, num_batches=50, device="cuda", n_top_vectors=n_p)
    _, _, eigenvectors_pattern = hessian_eig(model, loss_fn, data_loader_test_pattern, num_batches=50, device="cuda", n_top_vectors=n_p)
    _, _, eigenvectors_opacity_05 = hessian_eig(model, loss_fn, data_loader_05_test, num_batches=50, device="cuda", n_top_vectors=n_p)

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


def perturb_in_direction(fname, patterns_per_num, direction, n_p=50, just_return_proj_v=False, make_movie=False):
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

    _, _, eigenvectors_number = hessian_eig(model, loss_fn, data_loader_test_number, num_batches=50, device="cuda", n_top_vectors=n_p)
    _, _, eigenvectors_pattern = hessian_eig(model, loss_fn, data_loader_test_pattern, num_batches=50, device="cuda", n_top_vectors=n_p)

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


    t_values = np.linspace(0, 5.0, 15)

    # store results 
    acc_results = []
    loss_results = []
    for idx, t_val in enumerate(t_values):
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

        if make_movie:
            save_frames_weights(idx, model)

    # write results to textfile
    with open(f"perturbation_results_{direction}_acc.txt", "w") as f:
        for result in acc_results:
            f.write(f"{result[0]},{result[1]},{result[2]},{result[3]}\n")
    with open(f"perturbation_results_{direction}_loss.txt", "w") as f:
        for result in loss_results:
            f.write(f"{result[0]},{result[1]},{result[2]},{result[3]}\n")

    # plot results
    plot_pertubation_results(acc_results, f'perturbation_acc_results_{direction}.png', yaxis='Accuracy (%)')
    plot_pertubation_results(loss_results, f'perturbation_loss_results_{direction}.png', yaxis='Loss')

    make_conv_movies()

def get_hessian(eigenvectors, eigenvalues):
    eigenvectors = np.array(eigenvectors)
    eigenvalues = np.array(eigenvalues)
    assert len(eigenvectors) == len(eigenvalues), "The number of eigenvectors and eigenvalues must match"
    diag_matrix = np.diag(eigenvalues)
    approx_matrix = eigenvectors.T @ diag_matrix @ eigenvectors
    return approx_matrix

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
    _, eigenvalues_number, eigenvectors_number = hessian_eig(model, loss_fn, data_loader_test_number, num_batches=50, device="cuda", n_top_vectors=n)
    _, eigenvalues_pattern, eigenvectors_pattern = hessian_eig(model, loss_fn, data_loader_test_pattern, num_batches=50, device="cuda", n_top_vectors=n)
    _, eigenvalues_05, eigenvectors_05 = hessian_eig(model, loss_fn, data_loader_05_test, num_batches=50, device="cuda", n_top_vectors=n)

    # Get Hessian for each dataset
    H_number = get_hessian(eigenvectors_number, eigenvalues_number)
    H_pattern = get_hessian(eigenvectors_pattern, eigenvalues_pattern)
    H_05 = get_hessian(eigenvectors_05, eigenvalues_05)
    # Create file to store results
    with open(f"hessian_matrices_results_10patterns.txt", "w") as f:
        f.write(f"H_number:\n{H_number}\n\nH_pattern:\n{H_pattern}\n\nH_05:\n{H_05}\n")

def measure_loss_in_valley(fname="model_final_finetuned.ckpt", patterns_per_num=10, n=50):
    # Load model
    model = CNN(input_size=28)
    model.load_state_dict(t.load(fname))
    model.to(device="cuda")
    model.eval()
    loss_fn = t.nn.CrossEntropyLoss()
    data_loader_test_number, data_loader_test_pattern = load_pure_number_pattern_data(patterns_per_num, is_train=False)
    _, data_loader_05_test = load_mnist_data(patterns_per_num, opacity=0.5)
    _, eigenvalues_number, eigenvectors_number = hessian_eig(model, loss_fn, data_loader_test_number, num_batches=50, device="cuda", n_top_vectors=n)
    _, eigenvalues_pattern, eigenvectors_pattern = hessian_eig(model, loss_fn, data_loader_test_pattern, num_batches=50, device="cuda", n_top_vectors=n)
    _, eigenvalues_05, eigenvectors_05 = hessian_eig(model, loss_fn, data_loader_05_test, num_batches=50, device="cuda", n_top_vectors=n)

    H_05 = get_hessian(eigenvectors_05, eigenvalues_05)
    H_number = get_hessian(eigenvectors_number, eigenvalues_number)
    H_pattern = get_hessian(eigenvectors_pattern, eigenvalues_pattern)

    top_number = eigenvectors_number[-1]
    top_pattern = eigenvectors_pattern[-1]

    top_number_proj = np.matmul(orthogonal_complement(eigenvectors_pattern), top_number) 
    top_pattern_proj = np.matmul(orthogonal_complement(eigenvectors_number), top_pattern)

    # normalize
    top_number_proj = top_number_proj / np.linalg.norm(top_number_proj)
    top_pattern_proj = top_pattern_proj / np.linalg.norm(top_pattern_proj)

    H_op_number = top_number_proj.T @ H_05 @ top_number_proj
    H_op_pattern = top_pattern_proj.T @ H_05 @ top_pattern_proj

    print(f"H_op proj number: {H_op_number}")
    print(f"H_op proj pattern: {H_op_pattern}")

    H_number_number = top_number_proj.T @ H_number @ top_number_proj
    H_number_pattern = top_pattern_proj.T @ H_number @ top_pattern_proj

    print(f"H_number proj number: {H_number_number}")
    print(f"H_number proj pattern: {H_number_pattern}")

    H_pattern_number = top_number_proj.T @ H_pattern @ top_number_proj
    H_pattern_pattern = top_pattern_proj.T @ H_pattern @ top_pattern_proj

    print(f"H_pattern proj number: {H_pattern_number}")
    print(f"H_pattern proj pattern: {H_pattern_pattern}")

def test_linear_circuit_decomp(fname="model_final_finetuned.ckpt", patterns_per_num=10, n=50, top_vecs = 10):
    # Load model
    model = CNN(input_size=28)
    model.load_state_dict(t.load(fname))
    model.to(device="cuda")
    model.eval()
    loss_fn = t.nn.CrossEntropyLoss()
    data_loader_test_number, data_loader_test_pattern = load_pure_number_pattern_data(patterns_per_num, is_train=False)
    _, data_loader_05_test = load_mnist_data(patterns_per_num, opacity=0.5)
    _, _, eigenvectors_number = hessian_eig(model, loss_fn, data_loader_test_number, num_batches=50, device="cuda", n_top_vectors=n)
    _, _, eigenvectors_pattern = hessian_eig(model, loss_fn, data_loader_test_pattern, num_batches=50, device="cuda", n_top_vectors=n)
    _, _, eigenvectors_05 = hessian_eig(model, loss_fn, data_loader_05_test, num_batches=50, device="cuda", n_top_vectors=n)

    top_numbers = eigenvectors_number[-top_vecs:]
    top_patterns = eigenvectors_pattern[-top_vecs:]

    number_vector = [np.linalg.norm(np.matmul(orthogonal_complement(eigenvectors_05), v)) for v in top_numbers]
    pattern_vector = [np.linalg.norm(np.matmul(orthogonal_complement(eigenvectors_05), v)) for v in top_patterns]

    print(number_vector, pattern_vector)


def max_op_loss(fname, patterns_per_num, direction, n_p=50, just_return_proj_v=False):
    """
    fname: checkpoint file name
    patterns_per_num: number of patterns per digit
    direction: direction to preserve ('pattern' or 'number')
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
    _, data_loader_05_test = load_mnist_data(patterns_per_num, opacity=0.5)

    _, _, eigenvectors_number = hessian_eig(model, loss_fn, data_loader_test_number, num_batches=50, device="cuda", n_top_vectors=n_p)
    _, _, eigenvectors_pattern = hessian_eig(model, loss_fn, data_loader_test_pattern, num_batches=50, device="cuda", n_top_vectors=n_p)
    _, _, eigenvectors_opacity_05 = hessian_eig(model, loss_fn, data_loader_05_test, num_batches=50, device="cuda", n_top_vectors=n_p)

    if direction == 'number':
        orth = orthogonal_complement(eigenvectors_number) # 3340 x 3340
    elif direction == 'pattern':
        orth = orthogonal_complement(eigenvectors_pattern) # 3340 # 3340

    v = eigenvectors_opacity_05[-1] # 3340
    
    proj_v = np.matmul(orth, v) # 3340

    if just_return_proj_v:
        return proj_v

    # get opacity 0.5 dataloader
    _, data_loader_05_test = load_mnist_data(patterns_per_num, opacity=0.5)


    t_values = np.linspace(0, 10, 50)

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
    with open(f"perturbation_results_{direction}_acc_max_op.txt", "w") as f:
        for result in acc_results:
            f.write(f"{result[0]},{result[1]},{result[2]},{result[3]}\n")
    with open(f"perturbation_results_{direction}_loss_max_op.txt", "w") as f:
        for result in loss_results:
            f.write(f"{result[0]},{result[1]},{result[2]},{result[3]}\n")

    # plot results
    plot_pertubation_results(acc_results, f'perturbation_acc_results_max_op_min_{direction}.png', yaxis='Accuracy (%)')
    plot_pertubation_results(loss_results, f'perturbation_loss_results_max_op_min_{direction}.png', yaxis='Loss')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="path to model to load")
    parser.add_argument("--preserve", help="preserve number or pattern performance", type=str, default=None, required=False)
    parser.add_argument("--movie", help="generate movie visualizing conv kernels", action="store_true", default=False, required=False)
    parser.add_argument("--opacity", help="opacity of patterns in loss data", type=float, default=0.5, required=False)
    parser.add_argument("--patterns_per_num", help="number of patterns per digit", type=int, default=10, required=False)
    parser.add_argument("--mixed", help="use mixed data loader", action="store_true", default=False, required=False)
    args = parser.parse_args()
    model_path = args.model_path
    opacity = args.opacity
    patterns_per_num = args.patterns_per_num
    use_mixed_dataloader = args.mixed
    movie = args.movie
    if args.preserve is not None:
        perturb_in_direction(model_path, patterns_per_num, args.preserve, make_movie=movie)
    else:
        get_hessian_eig_mnist(model_path, patterns_per_num=patterns_per_num, opacity=opacity, use_mixed_dataloader=use_mixed_dataloader)