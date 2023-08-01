import numpy as np
import matplotlib.pyplot as plt
import torch as t

def get_submodule_vector_positions(model, get_module_parameters):
    """
    Function to get vector positions of a specified submodule's parameters.

    Args:
    model : a PyTorch model
    get_module_parameters : a function that given the model returns one of its submodules' parameters

    Returns:
    A dictionary with parameter names as keys and their positions in the vector as values, number of parameters in the full model
    """
    # Get the ids of the parameters of the submodule
    submodule_param_ids = set(id(p) for p in get_module_parameters(model))

    positions = {}
    start = 0
    n_params_full = 0
    for name, parameter in model.named_parameters():
        end = start + parameter.numel()  # numel() gives the total number of elements
        if id(parameter) in submodule_param_ids:  # Compare ids instead of tensors
            positions[name] = (start, end)
        start = end
        n_params_full += parameter.numel()

    return positions, n_params_full

def reshape_submodule_param_vector(model, get_module_parameters, vector):
    """
    Function to reshape a vector of parameters to the shape of the submodule's parameters.
    model : a PyTorch model
    get_module_parameters : a function that given the model returns one of its submodules' parameters
    vector : a vector of submodule parameters of size smaller than the full model's parameters
    """
    positions, n_params_full = get_submodule_vector_positions(model, get_module_parameters)
    reshaped_vector = t.zeros(size=(n_params_full,))
    idx = 0
    for _, (start, end) in positions.items():
        size = end - start
        reshaped_vector[start:end] = t.tensor(vector[idx:idx+size], dtype=t.float32)
        idx += size
    return reshaped_vector

def orthogonal_complement(eigenvector_matrix):
    """
    eigenvector_matrix: a matrix of eigenvectors (n_eigenvectors x n_weights)

    returns: a matrix of orthogonal vectors (n_weights x n_weights)
    """
    return np.eye(eigenvector_matrix.shape[-1]) - np.matmul(np.transpose(eigenvector_matrix), eigenvector_matrix)
    
def plot_pertubation_results(results, fname, yaxis):
    t_values = [result[0] for result in results]
    op_05_accuracy_values = [result[1] for result in results]
    pure_num_acc_values = [result[2] for result in results]
    pure_pattern_acc_values = [result[3] for result in results]
    plt.figure(figsize=(10, 6))

    plt.plot(t_values, op_05_accuracy_values, label='Opacity 0.5', color='r')
    plt.plot(t_values, pure_num_acc_values, label='Pure numbers', color='g')
    plt.plot(t_values, pure_pattern_acc_values, label='Pure patterns', color='b')

    plt.xlabel('Perturbation amount t')
    plt.ylabel(yaxis)
    plt.title(f'Model {yaxis} for different t values')
    plt.legend()
    plt.savefig(fname, format='png')
