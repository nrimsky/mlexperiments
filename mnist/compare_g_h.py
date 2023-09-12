from g_eig import hessian_eig_gauss_newton
import torch as t
from train_mnist import CNN, load_mnist_data
from hessian_eig import hessian_eig


def get_hessian_eig_mnist(fname, patterns_per_num, opacity=0.5):
    """
    Load model from fname checkpoint and calculate eigenvalues
    """
    model = CNN(input_size=28)
    model.load_state_dict(t.load(fname))
    model.to(device="cuda")
    model.eval()
    loss_fn = t.nn.CrossEntropyLoss()
    _, data_loader_test = load_mnist_data(patterns_per_num, opacity)
    _, heigenvalues, heigenvectors = hessian_eig(model, loss_fn, data_loader_test, num_batches=100, device="cuda", n_top_vectors=50)
    _, geigenvalues, geigenvectors = hessian_eig_gauss_newton(model, data_loader_test, num_batches=100, device="cuda", n_top_vectors=50)

    # Check similarity of eigenvectors
    # 1. Turn into tensors
    heigenvectors = t.tensor(heigenvectors) # n_top_vectors x num_params
    geigenvectors = t.tensor(geigenvectors)
    # 2. Normalize by dividing by norm
    heigenvectors /= heigenvectors.norm(dim=-1, keepdim=True)
    geigenvectors /= geigenvectors.norm(dim=-1, keepdim=True)
    # 3. Check similarity
    cos_sim = t.einsum("ij,ij->i", heigenvectors, geigenvectors)
    print("Cosine similarity of eigenvectors")
    print(cos_sim)
    # 4. Print as angle in degrees
    print("Angle in degrees")
    print(t.rad2deg(t.arccos(cos_sim)))

    # Check similarity of intra-eigenvalue ratios between Hessian and Gauss-Newton Hessian
    # 1. Sort eigenvalues
    heigenvalues = sorted(heigenvalues, reverse=True)
    geigenvalues = sorted(geigenvalues, reverse=True)
    # 2. Get ratios
    heigenvalue_ratios = [heigenvalues[i]/heigenvalues[i+1] for i in range(len(heigenvalues)-1)]
    geigenvalue_ratios = [geigenvalues[i]/geigenvalues[i+1] for i in range(len(geigenvalues)-1)]
    # 3. Check similarity
    print("Eigenvalue ratios")
    print("Hessian")
    print(heigenvalue_ratios)
    print("Gauss-Newton Hessian")
    print(geigenvalue_ratios)
    


if __name__ == "__main__":
    fname = "model_final_finetuned.ckpt"
    patterns_per_num = 10
    get_hessian_eig_mnist(fname, patterns_per_num)