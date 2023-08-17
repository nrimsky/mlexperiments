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

def get_weight_norm(model):
    return sum((p ** 2).sum() for p in model.parameters() if p.requires_grad)
