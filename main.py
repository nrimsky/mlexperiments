"""
Task:
Write a function that takes in a trained net, its training data and loss function.
Returns an estimate of the volume of the basin in which the trained parameter values ended up.
In particular, this should be the sort of function you could use to test the hypothesis that basin breadth accounts for generalization. 

This implementation uses the Gauss-Newton approximation to the Hessian of the loss function.
"""

import torch as t
from train_mnist import get_basin_calc_info_mnist
import numpy as np
import math

    
def calc_jacobian(model, data_loader_train, loss_fn, num_samples = 30):
    """
    Computes the Jacobian matrix (num_samples x num_parameters) of the model's loss on the training data with 
    respect to the model's parameters
    """
    rows = []
    samples = 0
    for _, (images, labels) in enumerate(data_loader_train):
        for image, label in zip(images, labels):
            samples += 1
            if samples > num_samples:
                break
            output = model(image.unsqueeze(0))
            # calculate the loss
            loss = loss_fn(output, label.unsqueeze(0))
            model.zero_grad()
            loss.backward()

            grads = t.cat([p.grad.view(-1) for p in model.parameters()])  # Flatten gradients
            rows.append(grads)

    return t.stack(rows)

def calculate_log_volume(T, hessian):
    """
    Calculates the logarithm of the volume of the basin of attraction based on a loss threshold.

    The volume of the basin of attraction is derived based on the eigenvalues of the Hessian of the loss 
    function. For each eigenvalue, a radius of the ellipsoid in the corresponding direction is calculated. 
    The volume of the ellipsoid is then calculated as the product of these radii times the volume of the 
    unit n-ball in n dimensions.

    The calculation is performed in the logarithmic domain to avoid potential overflow issues.

    Parameters:
    T (float): The loss threshold defining the basin of attraction.
    hessian (Tensor): The Hessian matrix of second derivatives of the loss function.

    Returns:
    float: The logarithm of the volume of the basin of attraction.
    """
    n = hessian.shape[0]  # Number of parameters in the model

    # Compute logarithms to avoid overflow
    log_Vn = n / 2.0 * np.log(np.pi) - math.lgamma(n / 2.0 + 1)  # volume of the unit n-ball in n dimensions

    # Eigenvalues
    eigenvalues_info = t.linalg.eig(hessian)
    eigenvalues = eigenvalues_info.eigenvalues.real  # We only need the real parts for symmetric matrix

    # Check for zero eigenvalues and replace them with a very small number to avoid zero determinant
    eigenvalues[eigenvalues == 0] = 1e-7  # Small positive value

    # Determinant calculation using the sum of logarithms of eigenvalues
    log_det_hessian = t.sum(t.log(eigenvalues.abs()))

    # Calculate the log volume
    log_volume = log_Vn + n / 2.0 * np.log(2 * T) - 0.5 * log_det_hessian

    return log_volume

def calc_basin_volume(model, loss, train_data_loader):
    jacobian = calc_jacobian(model, train_data_loader, loss)
    print(f"Jacobian shape: {jacobian.shape}")
    hessian = 2 * t.einsum('ij,ik->jk', jacobian, jacobian)  # Compute Hessian(Loss) with Gauss-Newton approximation
    log_v = calculate_log_volume(T=0.01, hessian=hessian)
    print("Basin Volume: exp({:.2f})".format(log_v))

if __name__ == "__main__":
    model, loss, train_data_loader = get_basin_calc_info_mnist()
    calc_basin_volume(model, loss, train_data_loader)