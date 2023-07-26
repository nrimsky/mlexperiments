import torch as t
from torch.autograd import grad
from scipy.sparse.linalg import LinearOperator, eigsh
from train_mnist import CNN, load_mnist_data, load_pure_number_pattern_data, CombinedDataLoader, test
import argparse
import numpy as np
from helpers import orthogonal_complement, plot_pertubation_results
from torch.nn.utils import parameters_to_vector, vector_to_parameters

def get_hessian_eigenvalues(model, loss_fn, train_data_loader, num_batches=30, device="cuda", n_top_vectors=100):
    """
    model: a pytorch model
    loss_fn: a pytorch loss function
    train_data_loader: a pytorch data loader
    num_batches: number of batches to use for the hessian calculation
    device: the device to use for the hessian calculation
    """

    num_params = sum(p.numel() for p in model.parameters())
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
        grad_params = grad(compute_loss(), model.parameters(), create_graph=True)
        flat_grad = t.cat([g.view(-1) for g in grad_params])
        grad_vector_product = t.sum(flat_grad * vector)
        hvp = grad(grad_vector_product, model.parameters(), retain_graph=True)
        return t.cat([g.contiguous().view(-1) for g in hvp])
    
    def matvec(v):
        v_tensor = t.tensor(v, dtype=t.float32, device=device)
        return hessian_vector_product(v_tensor).cpu().detach().numpy()
    
    linear_operator = LinearOperator((num_params, num_params), matvec=matvec)
    eigenvalues, eigenvectors = eigsh(linear_operator, k=200, tol=0.001, which='LM', return_eigenvectors=True)
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
    print(eigenvectors.shape)
    return tot, eigenvalues, eigenvectors[:n_top_vectors]


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

def perturb_in_direction(fname, patterns_per_num, direction, n_v=10, n_p=100):
    """
    fname: checkpoint file name
    patterns_per_num: number of patterns per digit
    direction: direction to perturb in ('pattern' or 'number')
    n_v: number of vectors to use for perturbation
    n_p: number of vectors to use for projection (n_v < n_p)
    """
    # Load model 
    model = CNN(input_size=28)
    model.load_state_dict(t.load(fname))
    model.to(device="cuda")
    model.eval()
    loss_fn = t.nn.CrossEntropyLoss()
    # Load pure pattern number data
    data_loader_test_number, data_loader_test_pattern = load_pure_number_pattern_data(patterns_per_num, is_train=False)
    _, eigenvalues_number, eigenvectors_number = get_hessian_eigenvalues(model, loss_fn, data_loader_test_number, num_batches=30, device="cuda", n_top_vectors=n_p)
    _, eigenvalues_pattern, eigenvectors_pattern = get_hessian_eigenvalues(model, loss_fn, data_loader_test_pattern, num_batches=30, device="cuda", n_top_vectors=n_p)
    if direction == 'number':
        orth = orthogonal_complement(eigenvectors_number) # 3340 x 3340
    elif direction == 'pattern':
        orth = orthogonal_complement(eigenvectors_pattern) # 3340 # 3340

    if direction == 'number':
        v = eigenvalues_pattern[:n_v] # n_v x 3340
    elif direction == 'pattern':
        v = eigenvalues_number[:n_v] # n_v x 3340
    
    proj_v = np.dot(orth, v) # n_v x 3340

    # get opacity 0.5 dataloader
    _, data_loader_05_test = load_mnist_data(patterns_per_num, opacity=0.5)

    # exp scale for t values 
    t_values = np.exp(np.linspace(-5, 5, 100))
    # store results 
    results = []
    for t_val in t_values:
        # load model 
        model = CNN(input_size=28)
        model.load_state_dict(t.load(fname))
        # perturb parameters by t * proj_v[0]
        params_vector = parameters_to_vector(model.parameters())
        params_vector += t_val * proj_v[0]
        vector_to_parameters(params_vector, model.parameters())
        # move model to cuda
        model.to(device="cuda")
        # evaluating model 
        op_05_accuracy = test(model, data_loader_05_test, do_print=False)
        pure_num_acc = test(model, data_loader_test_number, do_print=False)
        pure_pattern_acc = test(model, data_loader_test_pattern, do_print=False)
        # print results
        print(f"t_val: {t_val:.2f}, direction: {direction}, op_05_acc: {op_05_accuracy:.2f}, pure_num_acc: {pure_num_acc:.2f}, pure_pattern_acc: {pure_pattern_acc:.2f}")
        # store results
        results.append((t_val, op_05_accuracy, pure_num_acc, pure_pattern_acc))
    # plot results
    plot_pertubation_results(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version", help="version of model to load")
    parser.add_argument("--perturb", help="perturb in direction of number or pattern", type=str, default=None, required=False)
    parser.add_argument("--opacity", help="opacity of patterns in loss data", type=float, default=0.5, required=False)
    parser.add_argument("--patterns_per_num", help="number of patterns per digit", type=int, default=10, required=False)
    parser.add_argument("--mixed", help="use mixed data loader", action="store_true", default=False, required=False)
    args = parser.parse_args()
    version = args.version
    opacity = args.opacity
    patterns_per_num = args.patterns_per_num
    use_mixed_dataloader = args.mixed
    if args.perturb is not None:
        perturb_in_direction(f"./models/model_{version}.ckpt", patterns_per_num, args.perturb)
    else:
        get_hessian_eig_mnist(f"./models/model_{version}.ckpt", patterns_per_num=patterns_per_num, opacity=opacity, use_mixed_dataloader=use_mixed_dataloader)

    # @ opacity 0.5
    # 0.0: 171, 52, 32, 9
    # 0.2: 56, 15, 10, 6
    # 0.4: 32, 9, 9, 3
    # 0.6: 35, 10, 9, 2
    # 0.7: 35, 9, 9, 2
    # 0.8: 39, 10, 9, 2
    # 0.9: 40, 11, 9, 3
    # 0.95: 46, 11, 9, 5
    # 1.0: 47, 11, 9, 4

    # Final finetuned 28, 9, 7, 1

    # 1.0 @ opacity 0 (pure num): 174, 38, 20, 9
    # 1.0 @ opacity 1 (pure patterns): 14, 4, 0, 0
    # 1.0 @ 0/1 combination: 109, 20, 10, 7

    # Mixture pure patterns numbers 62, 14, 9, 5

    # Mixture pure patterns numbers 10d pattern @ 1.0 - 145, 52, 29, 9
    # Mixture pure patterns numbers 10d pattern @ 0.0 - 161, 34, 16, 8
    # Mixture pure patterns numbers 10d pattern @ 0/1 combination - 192, 45, 23, 8
    # Mixture pure patterns numbers 10d pattern @ opacity 0.5 - 178, 70, 44, 9

    # Direct 0.5 opacity trained 10d pattern @ 1.0 - 183, 144, 120, 61 // 161, 131, 103, 58
    # Direct 0.5 opacity trained 10d pattern @ 0.0 - 181, 53, 30, 9
    # Direct 0.5 opacity trained 10d pattern @ 0/1 combination - 187, 140, 109, 49
    # Direct 0.5 opacity trained 10d pattern @ opacity 0.5 - 189, 68, 39, 9

    # Direct 0.5 opacity trained 51, 12, 10, 6




# python hessian_eig.py direct_0.5_ppn_10 --opacity=0.0 --patterns_per_num=10
# Number of eigenvalues greater than 0.1: 153
# Number of eigenvalues greater than 1: 78
# Number of eigenvalues greater than 2: 49
# Number of eigenvalues greater than 10: 16

# python hessian_eig.py direct_0.5_ppn_10 --opacity=0.5 --patterns_per_num=10
# Number of eigenvalues greater than 0.1: 185
# Number of eigenvalues greater than 1: 74
# Number of eigenvalues greater than 2: 44
# Number of eigenvalues greater than 10: 10