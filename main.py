import torch as t
from torch.autograd.functional import hvp
import numpy as np
from math import lgamma
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from scipy.sparse.linalg import LinearOperator, eigsh
from train_mnist import get_basin_calc_info_mnist
import copy


def calculate_log_volume(T, eigenvalues, n):
    log_Vn = n / 2.0 * np.log(np.pi) - lgamma(n / 2.0 + 1)
    eigenvalues = np.clip(eigenvalues, 1e-7, None)
    log_det_hessian = np.sum(np.log(eigenvalues))
    log_volume = log_Vn + n / 2.0 * np.log(2 * T) - 0.5 * log_det_hessian
    return log_volume


def calc_basin_volume(model, loss_fn, train_data_loader, num_batches=20):
    num_params = sum(p.numel() for p in model.parameters())
    subset_images, subset_labels = [], []
    for batch_idx, (images, labels) in enumerate(train_data_loader):
        if batch_idx >= num_batches:
            break
        subset_images.append(images)
        subset_labels.append(labels)
    subset_images = t.cat(subset_images)
    subset_labels = t.cat(subset_labels)

    def hvp_fn(v):
        v_tensor = t.tensor(v, dtype=next(model.parameters()).dtype)
        model_clone = copy.deepcopy(model)  # Create a clone of the model
        vector_to_parameters(v_tensor, model_clone.parameters())  # Work on the clone instead

        def compute_loss(p):
            vector_to_parameters(p, model_clone.parameters())
            outputs = model_clone(subset_images)
            return loss_fn(outputs, subset_labels)

        hv = hvp(compute_loss, parameters_to_vector(model_clone.parameters()), v_tensor)[1]
        return hv.view(-1).detach().numpy()  # Convert back to numpy

    H = LinearOperator((num_params, num_params), matvec=hvp_fn)
    eigenvalues = eigsh(H, return_eigenvectors=False)
    vol = calculate_log_volume(T=80, eigenvalues=eigenvalues, n=num_params)
    print(f"Log Basin Volume: {vol}")


if __name__ == "__main__":
    model, loss, train_data_loader = get_basin_calc_info_mnist()
    calc_basin_volume(model, loss, train_data_loader)
