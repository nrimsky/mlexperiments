import torch as t
from torch.autograd import grad
from scipy.sparse.linalg import LinearOperator, eigsh
import numpy as np
from utils import get_weight_norm

def hessian_eig_modular_addition(
    model,
    loss_fn,
    train_data_loader,
    device="cuda",
    n_top_vectors=200,
    param_extract_fn=None,
    reg=0.002,
):
    param_extract_fn = param_extract_fn or (lambda x: x.parameters())
    num_params = sum(p.numel() for p in param_extract_fn(model))
    subset_a, subset_b, subset_res = [], [], []
    for a, b, res in train_data_loader:
        subset_a.append(a.to(device))
        subset_b.append(b.to(device))
        subset_res.append(res.to(device))
    subset_a = t.cat(subset_a)
    subset_b = t.cat(subset_b)
    subset_res = t.cat(subset_res)

    def compute_loss():
        output = model(subset_a, subset_b)
        return loss_fn(output, subset_res) + reg * get_weight_norm(
            model
        )  # hacky way to add weight norm

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
    eigenvalues, eigenvectors = eigsh(
        linear_operator,
        k=n_top_vectors,
        tol=0.001,
        which="LM",
        return_eigenvectors=True,
    )
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
