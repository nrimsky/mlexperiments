import torch as t
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from utils import reshape_submodule_param_vector, get_weight_norm
from mlp_modular import (
    test_model,
    get_train_test_loaders,
    MLP_unchunked,
)
import torch as t
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from hessian_eig import hessian_eig_modular_addition, hessian_comparison
from generate_movie import (
    plot_embeddings_movie_unchunked,
    plot_embeddings_movie,
    run_movie_cmd,
)


def get_module_parameters(model):
    return model.parameters()


def quadratic_form(top_eigenvalues, top_eigenvectors, v):
    # Create a diagonal matrix with the eigenvalues
    top_eigenvectors = t.tensor(top_eigenvectors, dtype = t.float32)
    top_eigenvalues = t.tensor(top_eigenvalues, dtype = t.float32)
    Lambda = t.diag(top_eigenvalues)
    Q = top_eigenvectors.T
    # Return the function that evaluates the quadratic form
    intermediate = t.matmul(Q.T, v)  # Q^T * v
    scaled = intermediate * top_eigenvalues  # Element-wise multiplication with Lambda
    result = t.matmul(Q, scaled)  # Q * scaled
    evaluated = t.dot(v, result)  # v^T * result
    return evaluated



def sphere_localized_loss_adjustment(
    model,
    top_eigenvectors,
    offset,
    top_eigenvalues = None,    
    radius=1,
    lambda_sphere=10,
    lambda_orth=0.1,
    lambda_stab = 0,
    device="cuda",
    subtract_quad = False,
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
    r_proj_params = t.norm(params_proj - offset_proj)
    sphere_reg = lambda_sphere * (r_proj_params - radius) ** 2
    stab_reg = lambda_stab * (r_proj_params - radius) ** 4
    quad_term = 0
    if not top_eigenvalues is None:
        quad_term = quadratic_form(top_eigenvalues, top_eigenvectors, params_proj)
    orth_reg = (
        lambda_orth * t.norm(params_vector - offset - params_proj + offset_proj) ** 2
    )
    return sphere_reg, orth_reg, stab_reg, quad_term


def train_in_sphere(
    model,
    dataloader,
    top_eigenvectors,
    top_eigenvalues = None,
    subtract_quad = False, 
    radius=1,
    lambda_sphere=10,
    lambda_orth=0.1,
    lambda_stab=0, #4th-power term to neutralize runaway SGD after subtracting quadratic term
    lr=1e-3,
    n_epochs=3,
    device="cuda",
    lr_decay=0.999,
    weight_reg=0.05,
    initial_params=None,
    frame_idx_start=0,
):
    model.to(device)
    offset = parameters_to_vector(model.parameters()).detach()
    if initial_params is None:
        # reshape all eigenvectors to be the same shape as the model parameters
        top_eigenvectors = t.stack(
            [
                reshape_submodule_param_vector(model, get_module_parameters, v)
                for v in top_eigenvectors
            ]
        )
        # Adjust model weights to be on the sphere of high eigenvectors
        n_eigenvectors = top_eigenvectors.shape[0]
        rand_vec = t.randn(n_eigenvectors)
        unit_sphere_vec = rand_vec @ top_eigenvectors
        unit_sphere_vec /= t.norm(unit_sphere_vec)
        point_on_sphere = offset.to(device) + radius * unit_sphere_vec.to(device)
        # load the point on the sphere into the model
        vector_to_parameters(point_on_sphere, model.parameters())
    else:
        vector_to_parameters(initial_params, model.parameters())

    optimizer = t.optim.AdamW(get_module_parameters(model), lr=lr, weight_decay=0)
    scheduler = t.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    ce_loss = t.nn.CrossEntropyLoss()
    model.train()
    idx = 0
    tot_sphere_reg = 0
    tot_orth_reg = 0
    tot_ce_loss = 0
    tot_weight_reg_loss = 0
    step = frame_idx_start
    for epoch in range(n_epochs):
        for a, b, res in dataloader:
            if idx % 100 == 0:
                step += 1
                plot_embeddings_movie_unchunked(model, step)
            idx += 1
            optimizer.zero_grad()
            sphere_reg, orth_reg, stab_reg, quad_term = sphere_localized_loss_adjustment(
                model,
                top_eigenvectors,
                offset,
                radius = radius,
                lambda_sphere = lambda_sphere,
                lambda_orth = lambda_orth,
                top_eigenvalues = top_eigenvalues,
                subtract_quad = subtract_quad,
                device = device,
            )
            loss_main = ce_loss(model(a.to(device), b.to(device)), res.to(device))
            weight_reg_loss = get_weight_norm(model) * weight_reg
            tot_sphere_reg += sphere_reg.item()
            tot_orth_reg += orth_reg.item()
            tot_ce_loss += loss_main.item()
            tot_weight_reg_loss += float(weight_reg_loss)
            loss = loss_main + sphere_reg + orth_reg + weight_reg + stab_reg - quad_term
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch % 50 == 0:
            print(
                f"Epoch {epoch}/{n_epochs}, avg_sphere_reg = {tot_sphere_reg/idx}, avg_orth_reg = {tot_orth_reg/idx}, avg_ce_loss = {tot_ce_loss/idx} avg_weight_reg_loss = {tot_weight_reg_loss/idx}"
            )
            idx = 0
            tot_sphere_reg = 0
            tot_orth_reg = 0
            tot_ce_loss = 0
            tot_weight_reg_loss = 0
    val_loss, val_acc = test_model(model, dataloader, device=device, criterion=ce_loss)
    print(f"Final validation loss: {val_loss}, final validation accuracy: {val_acc}")
    t.save(model.state_dict(), "modular_addition_sphere_model.pth")
    return model, step



def main(checkpoint_path="modular_addition.ckpt"):
    # Parameters
    DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'
    VOCAB_SIZE = 38
    EMBED_DIM = 14
    HIDDEN_DIM = 32
    N_EPOCHS = 3000
    N_EIGENVECTORS = 35
    BOUND_EIGENVECTORS = 20
    LAMBDA_SPHERE = 20
    LAMBDA_ORTH = 0.5
    LAMBDA_STAB = 0.1
    SUBTRACT_QUAD = True
    LR = 0.01
    LR_DECAY = 0.9997
    WEIGHT_REG = 0.001
    # Used to calculate eigenvectors for sphere search
    model = MLP_unchunked(
        vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM
    )
    model.load_state_dict(t.load(checkpoint_path))
    model.to(device=DEVICE)
    model.eval()

    loss_fn = t.nn.CrossEntropyLoss()
    train_loader, test_loader = get_train_test_loaders(
        train_frac=0.4, batch_size=256, vocab_size=VOCAB_SIZE
    )
    _, eigenvalues, eigenvectors = hessian_eig_modular_addition(
        model,
        loss_fn,
        test_loader,
        device = DEVICE,
        n_top_vectors=N_EIGENVECTORS,
        param_extract_fn=get_module_parameters,
        reg=WEIGHT_REG,
    )
    eigenvalues, eigenvectors = eigenvalues[:-BOUND_EIGENVECTORS], eigenvectors[:-BOUND_EIGENVECTORS]
    initial_params = None
    frame_idx = 0
    for circuit in range(1):
        if circuit != 0:
            print("Using previous model to calculate eigenvectors")
            eigen_model = MLP_unchunked(
                vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM
            )
            eigen_model.load_state_dict(t.load("modular_addition_sphere_model.pth"))
            eigen_model.to(device)
            eigen_model.eval()
            _, eigenvalues, eigenvectors = hessian_eig_modular_addition(
                eigen_model,
                loss_fn,
                test_loader,
                device = DEVICE,
                n_top_vectors=N_EIGENVECTORS,
                param_extract_fn=get_module_parameters,
                reg=WEIGHT_REG,
            )
        for radius in [15, 25, 30, 35, 40]:
            _, frame_idx = train_in_sphere(
                model,
                train_loader,
                eigenvectors,
                top_eigenvalues = eigenvalues,
                radius=radius,
                subtract_quad = SUBTRACT_QUAD,
                lambda_sphere = LAMBDA_SPHERE,
                lambda_stab = LAMBDA_STAB,
                lambda_orth = LAMBDA_ORTH,
                lr=LR,
                n_epochs=N_EPOCHS,
                device = DEVICE,
                lr_decay=LR_DECAY,
                weight_reg=WEIGHT_REG,
                initial_params=initial_params,
                frame_idx_start=frame_idx,
            )
            model.project_to_fourier_mode(f"modular_addition_sphere_{radius}.png")
            start_model = MLP_unchunked(
                vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM
            )
            start_model.load_state_dict(t.load("modular_addition_sphere_model.pth"))
            initial_params = (
                parameters_to_vector(start_model.parameters()).detach().to(DEVICE)
            )
        run_movie_cmd()


        def ablate_other_modes(modes):


if __name__ == "__main__":
    main()
