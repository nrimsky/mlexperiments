import torch as t
import numpy as np
from tqdm import tqdm

def get_model_output_loss(parameters):
    return parameters[0] ** 2 + 3 * parameters[1] ** 2 + 5 * parameters[0] * parameters[1]

def finite_diff_hessian(func, x, epsilon=1e-4):
    x = x.detach().numpy()
    # Dimension of the input
    dim = len(x)
    
    # Initialize the Hessian matrix
    hessian = np.zeros((dim, dim))

    f_x = func(x)

    def f_vv(x, v):
        x_v_plus = np.copy(x)
        x_v_minus = np.copy(x)
        x_v_plus[v] += epsilon
        x_v_minus[v] -= epsilon
        return func(x_v_plus) - 2*f_x + func(x_v_minus) / (2 * (epsilon**2))
    
    # Compute the diagonal elements
    for i in tqdm(range(dim)):        
        hessian[i, i] = f_vv(x, i)

    print("Diagonal elements computed")
    
    # Compute the off-diagonal elements
    for i in tqdm(range(dim)):
        for j in range(i+1, dim):
            hessian[i, j] = (f_vv(x, [i, j]) - hessian[i, i] - hessian[j, j])/2
            hessian[j, i] = hessian[i, j]  # the Hessian matrix is symmetric
            
    return hessian

if __name__ == "__main__":
    h = finite_diff_hessian(get_model_output_loss, t.tensor([0.0, 0.0]))
    print(h)