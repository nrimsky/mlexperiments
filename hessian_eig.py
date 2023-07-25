import torch as t
from torch.autograd import grad
from scipy.sparse.linalg import LinearOperator, eigsh
from train_mnist import get_basin_calc_info_mnist


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
    threshold = 0.1
    for e in eigenvalues:
        print("{:.2f}".format(e))
        if e > threshold:
            tot += 1
    print(f"Number of eigenvalues greater than {threshold}: {tot}")
    return tot, eigenvalues

