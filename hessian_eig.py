import torch as t
from torch.autograd import grad
from scipy.sparse.linalg import LinearOperator, eigsh
from train_mnist import CNN, load_mnist_data, load_pure_number_pattern_data, CombinedDataLoader
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


def get_hessian_eig_mnist(fname, opacity=0.5, use_mixed_dataloader=False):
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
    if use_mixed_dataloader:
        d1, d2 = load_pure_number_pattern_data(patterns, is_train=False)
        data_loader_test = CombinedDataLoader(d1, d2)
    get_hessian_eigenvalues(model, loss_fn, data_loader_test, num_batches=30, device="cuda")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version", help="version of model to load")
    parser.add_argument("--opacity", help="opacity of patterns in loss data", type=float, default=0.5, required=False)
    parser.add_argument("--mixed", help="use mixed data loader", action="store_true", default=False, required=False)
    args = parser.parse_args()
    version = args.version
    opacity = args.opacity
    use_mixed_dataloader = args.mixed
    get_hessian_eig_mnist(f"./models/model_{version}.ckpt", opacity=opacity, use_mixed_dataloader=use_mixed_dataloader)

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





