import torch as t
from torch.autograd import grad
from scipy.sparse.linalg import LinearOperator, eigsh
from train_mnist import CNN, load_mnist_data
import argparse

def get_hessian_eigenvalues(model, loss_fn, train_data_loader, num_batches=30, device="cuda"):
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
    eigenvalues, _ = eigsh(linear_operator, k=200, tol=0.001, which='LM')
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
    return tot, eigenvalues


def get_hessian_eig_mnist(fname, opacity=0.5):
    """
    Load model from fname checkpoint and calculate eigenvalues
    """
    patterns = t.load("./patterns.pt")
    model = CNN(input_size=28)
    model.load_state_dict(t.load(fname))
    model.to(device="cuda")
    model.eval()
    loss_fn = t.nn.CrossEntropyLoss()
    _, data_loader_test = load_mnist_data(patterns, opacity)
    get_hessian_eigenvalues(model, loss_fn, data_loader_test, num_batches=30, device="cuda")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version", help="version of model to load")
    args = parser.parse_args()
    version = args.version
    get_hessian_eig_mnist(f"./models/model_{version}.ckpt")
    # 0.0: 184, 72, 33, 9 / 180, 47, 25, 9 (89%)
    # 0.2:                / 180, 38, 21, 9 (91%)
    # 0.4:                / 184, 28, 18, 9 (92%)
    # 0.6: 181, 44, 20, 9 / 181, 28, 17, 9 (90%)
    # 0.7:                / 165, 25, 14, 9 (87%)
    # 0.8: 176, 32, 15, 7 / 158, 23, 15, 6 (76%)
    # 0.9: 171, 21, 12, 5 / 126, 20, 10, 3(50%)
    # 0.95: 131, 15, 10, 1 / 94, 12, 10, 1 (37%)
    # 1.0: 103, 13, 5, 0 / 83, 13, 4, 0 (27%)

