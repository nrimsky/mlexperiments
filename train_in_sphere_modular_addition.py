import torch as t
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from helpers import reshape_submodule_param_vector
from mlp_modular import (
    MLP,
    test_model,
    get_train_test_loaders,
    plot_embeddings,
    plot_embeddings_chunks,
)
import torch as t
from torch.autograd import grad
from scipy.sparse.linalg import LinearOperator, eigsh
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def hessian_eig(
    model,
    loss_fn,
    train_data_loader,
    num_batches=30,
    device="cuda",
    n_top_vectors=200,
    param_extract_fn=None,
):
    param_extract_fn = param_extract_fn or (lambda x: x.parameters())
    num_params = sum(p.numel() for p in param_extract_fn(model))
    subset_a, subset_b, subset_res = [], [], []
    for batch_idx, (a, b, res) in enumerate(train_data_loader):
        if batch_idx >= num_batches:
            break
        subset_a.append(a.to(device))
        subset_b.append(b.to(device))
        subset_res.append(res.to(device))
    subset_a = t.cat(subset_a)
    subset_b = t.cat(subset_b)
    subset_res = t.cat(subset_res)

    def compute_loss():
        output = model(subset_a, subset_b)
        return loss_fn(output, subset_res)

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
        linear_operator, k=n_top_vectors, tol=0.1, which="LM", return_eigenvectors=True
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


def get_module_parameters(model):
    return model.parameters()


def sphere_localized_loss_adjustment(
    model,
    top_eigenvectors,
    offset,
    radius=1,
    lambda_sphere=10,
    lambda_orth=0.1,
    device="cuda",
):
    """
    model: mlp
    top_eigenvectors: top eigenvectors of the Hessian on the mixed data problem [n_eigenvectors x n_params] (arranged in small-to-large order)
    offset: weight vector where we start
    radius: radius of the sphere we search on
    lambda_sphere: how much to penalize being outside the sphere
    lambda_orth: how much to penalize being inside the orthogonal complement of the top eigenvectors
    """
    top_eigenvectors = t.tensor(top_eigenvectors).to(device)
    offset = offset.to(device)
    proj_matrix = t.mm(
        top_eigenvectors.T, top_eigenvectors
    )  # (n_params x n_eigenvectors) @ (n_eigenvectors x n_params) -> (n_params x n_params)
    params_vector = parameters_to_vector(model.parameters())
    params_proj = t.mv(proj_matrix, params_vector)
    offset_proj = t.mv(proj_matrix, offset)
    r_proj_params = t.sqrt(t.norm(params_proj - offset_proj))
    sphere_reg = lambda_sphere * (r_proj_params - radius) ** 2
    orth_reg = lambda_orth * t.norm(params_vector - offset - params_proj + offset_proj)
    return sphere_reg, orth_reg


def train_in_sphere(
    model,
    dataloader,
    top_eigenvectors,
    radius=1,
    lambda_sphere=10,
    lambda_orth=0.1,
    lr=1e-3,
    n_epochs=3,
    device="cuda",
    lr_decay=0.999,
    weight_decay=0.05,
):
    model.to(device)
    offset = parameters_to_vector(model.parameters()).detach()
    optimizer = t.optim.AdamW(
        get_module_parameters(model), lr=lr, weight_decay=weight_decay
    )
    # reshape all eigenvectors to be the same shape as the model parameters
    top_eigenvectors = t.stack(
        [
            reshape_submodule_param_vector(model, get_module_parameters, v)
            for v in top_eigenvectors
        ]
    )
    scheduler = t.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    loss_fn = t.nn.CrossEntropyLoss()
    # Adjust model weights to be on the sphere of high eigenvectors
    n_eigenvectors = top_eigenvectors.shape[0]
    rand_vec = t.randn(n_eigenvectors)
    unit_sphere_vec = rand_vec @ top_eigenvectors
    unit_sphere_vec /= t.norm(unit_sphere_vec)
    point_on_sphere = offset.to(device) + radius * unit_sphere_vec.to(device)
    # load the point on the sphere into the model
    vector_to_parameters(point_on_sphere, model.parameters())
    model.train()
    idx = 0
    tot_sphere_reg = 0
    tot_orth_reg = 0
    tot_ce_loss = 0
    for epoch in range(n_epochs):
        for a, b, res in dataloader:
            idx += 1
            optimizer.zero_grad()
            sphere_reg, orth_reg = sphere_localized_loss_adjustment(
                model,
                top_eigenvectors,
                offset,
                radius,
                lambda_sphere,
                lambda_orth,
                device=device,
            )
            loss_main = loss_fn(model(a.to(device), b.to(device)), res.to(device))
            tot_sphere_reg += sphere_reg.item()
            tot_orth_reg += orth_reg.item()
            tot_ce_loss += loss_main.item()
            loss = loss_main + sphere_reg + orth_reg
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch % 50 == 0:
            print(
                f"Epoch {epoch}/{n_epochs}, avg_sphere_reg = {tot_sphere_reg/idx}, avg_orth_reg = {tot_orth_reg/idx}, avg_ce_loss = {tot_ce_loss/idx}"
            )
            idx = 0
            tot_sphere_reg = 0
            tot_orth_reg = 0
            tot_ce_loss = 0
    val_loss, val_acc = test_model(model, dataloader, device=device, criterion=loss_fn)
    print(f"Final validation loss: {val_loss}, final validation accuracy: {val_acc}")
    t.save(model.state_dict(), "modular_addition_sphere_model.pth")
    return model


def main():
    # Parameters
    VOCAB_SIZE = 114
    EMBED_DIM = 14
    HIDDEN_DIM = 8
    CHECKPOINT_PATH = "modular_addition.ckpt"
    N_EPOCHS = 5000
    N_EIGENVECTORS = 8
    RADIUS = 20
    LAMBDA_SPHERE = 10
    LAMBDA_ORTH = 20
    LR = 0.02
    LR_DECAY = 0.999
    WEIGHT_DECAY = 0.05
    # Main code
    model = MLP(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM)
    model.load_state_dict(t.load(CHECKPOINT_PATH))
    model.to(device="cuda")
    model.eval()
    loss_fn = t.nn.CrossEntropyLoss()
    train_loader, test_loader = get_train_test_loaders(
        train_frac=0.4, batch_size=256, vocab_size=VOCAB_SIZE
    )
    _, _, eigenvectors = hessian_eig(
        model,
        loss_fn,
        test_loader,
        device="cuda",
        n_top_vectors=N_EIGENVECTORS,
        param_extract_fn=get_module_parameters,
    )
    train_in_sphere(
        model,
        train_loader,
        eigenvectors,
        radius=RADIUS,
        lambda_sphere=LAMBDA_SPHERE,
        lambda_orth=LAMBDA_ORTH,
        lr=LR,
        n_epochs=N_EPOCHS,
        device="cuda",
        lr_decay=LR_DECAY,
        weight_decay=WEIGHT_DECAY,
    )
    plot_embeddings(model, VOCAB_SIZE)
    plot_embeddings_chunks(model)


if __name__ == "__main__":
    main()
