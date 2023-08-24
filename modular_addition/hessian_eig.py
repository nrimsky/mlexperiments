import torch as t
from torch.autograd import grad
from scipy.sparse.linalg import LinearOperator, eigsh
import numpy as np
from utils import get_weight_norm
from mlp_modular import MLP_unchunked, get_train_test_loaders


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


# comparing generalization spaces between "base" model and "sphere search" model
def hessian_comparison(model, new_model, loss_fn, train_data_loader, reg):
    new_eigenvectors = hessian_eig_modular_addition(
        new_model,
        loss_fn,
        train_data_loader,
        device="cuda",
        n_top_vectors=15,
        param_extract_fn=None,
        reg=reg,
    )[2]
    old_eigenvectors = hessian_eig_modular_addition(
        model,
        loss_fn,
        train_data_loader,
        device="cuda",
        n_top_vectors=5,
        param_extract_fn=None,
        reg=reg,
    )[2]

    new_eigenvectors = t.tensor(new_eigenvectors, dtype=t.float32, device="cuda")
    old_eigenvectors = t.tensor(old_eigenvectors, dtype=t.float32, device="cuda")
    new_proj = t.mm(new_eigenvectors.T, new_eigenvectors)
    old_projected = t.mm(new_proj, old_eigenvectors.T)
    defect = t.norm(old_eigenvectors, p=2) / t.norm(old_projected, p=2)
    return defect.item()


if __name__ == "__main__":
    vocab_size = 38
    embed_dim = 14
    hidden_dim = 32
    reg = 0.001
    model1 = MLP_unchunked(embed_dim, vocab_size, hidden_dim)
    model2 = MLP_unchunked(embed_dim, vocab_size, hidden_dim)
    model1.to("cuda")
    model2.to("cuda")
    model1.load_state_dict(t.load("modular_addition.ckpt"))
    model2.load_state_dict(t.load("modular_addition_sphere_model.pth"))
    loss = t.nn.CrossEntropyLoss()
    train_loader, _ = get_train_test_loaders(1, 64, vocab_size)
    print(hessian_comparison(model1, model2, loss, train_loader, reg))
