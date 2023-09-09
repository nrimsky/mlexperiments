import torch as t
from torch.autograd import grad
import numpy as np


def hessian_eig_gauss_newton(
    model, train_data_loader, num_batches=30, device="cuda", n_top_vectors=200
):
    """
    Get eigenvalues and eigenvectors of the Gauss-Newton Hessian for a model trained with cross-entropy loss.
    $H= \expectation[J^T H J]$.
    $J is the network's parameter-output Jacobian
    $H$ is the Hessian of the loss with respect to the network's outputs, and the expectation is with respect to the empirical distribution.
    The GNH can be seen as an approximation to the Hessian which linearizes the network's parameter-output mapping around the current parameters.
    """
    model = model.to(device)
    model.eval()  # Set to eval mode to avoid unnecessary updates

    # Create a subset of the training data
    subset_images, subset_labels = [], []
    for batch_idx, (images, labels) in enumerate(train_data_loader):
        if batch_idx >= num_batches:
            break
        subset_images.append(images.to(device))
        subset_labels.append(labels.to(device))
    subset_images = t.cat(subset_images, 0)
    subset_labels = t.cat(subset_labels, 0)

    model.zero_grad()
    outputs = model(subset_images)

    # Compute the Jacobian
    jacobian_elems = []
    for j in range(outputs.shape[1]):
        grad_params = grad(outputs[:, j].sum(), model.parameters(), retain_graph=True)
        jacobian_elems.append(t.cat([g.view(-1) for g in grad_params]))
    jacobian = t.stack(jacobian_elems, dim=0)

    # H for cross-entropy loss
    softmax_outputs = t.softmax(outputs, dim=1)
    batch_size, num_classes = softmax_outputs.shape
    I = t.eye(num_classes).to(device)

    out_hessian_diags = softmax_outputs * (1 - softmax_outputs)
    out_hessian_off_diags = -softmax_outputs.unsqueeze(2) * softmax_outputs.unsqueeze(1)

    out_hessians = (
        I.unsqueeze(0) * out_hessian_diags.unsqueeze(-1)
        + (1 - I).unsqueeze(0) * out_hessian_off_diags
    )

    # Compute the Gauss-Newton Hessian
    E_JtHJ = t.einsum(
        "bni,bnn,bnj->ij", jacobian.unsqueeze(0), out_hessians, jacobian.unsqueeze(0)
    )
    E_JtHJ /= batch_size

    # Compute the top eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(E_JtHJ.cpu().detach().numpy())

    # Sort eigenvalues and eigenvectors
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Print some statistics
    print("\n".join("{:.2f}".format(e) for e in eigenvalues[-n_top_vectors:]))
    for threshold in [0.1, 1, 2, 10]:
        tot = (eigenvalues > threshold).sum()
        print(f"Number of eigenvalues greater than {threshold}: {tot}")

    return 0, eigenvalues[-n_top_vectors:], eigenvectors[:, -n_top_vectors:].T
