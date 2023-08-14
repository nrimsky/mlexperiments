import torch as t
from torch.autograd import grad
from scipy.sparse.linalg import LinearOperator, eigsh
import numpy as np

def get_hessian_eigenvalues(model, loss_fn, train_data_loader, num_batches, device, n_top_vectors, param_extract_fn):
    """
    model: a pytorch model
    loss_fn: a pytorch loss function
    train_data_loader: a pytorch data loader
    num_batches: number of batches to use for the hessian calculation
    device: the device to use for the hessian calculation
    n_top_vectors: number of top eigenvalues / eigenvectors to return
    param_extract_fn: a function that takes a model and returns a list of parameters to compute the hessian with respect to (pass None to use all parameters)

    returns: a tuple of (eigenvalues, eigenvectors)
    eigenvalues: a numpy array of the top eigenvalues, arranged in increasing order
    eigenvectors: a numpy array of the top eigenvectors, arranged in increasing order, shape (n_top_vectors, num_params)
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
    eigenvectors = np.transpose(eigenvectors)
    return eigenvalues, eigenvectors