import torch.nn as nn
import torch.nn.functional as F
import torch as t
from torchvision import datasets, transforms
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from random import randint
from torch.utils.data import Dataset, DataLoader
from glob import glob
import numpy as np
from train_mnist import CNN, load_mnist_data, load_pure_number_pattern_data, CombinedDataLoader, test


def sphere_localized_loss(model, loss, matrix, offset, lambda_sphere = 10, lambda_orth = .1):
 	"""docstring for sphere_localize"""
    num_params, num_evectors = matrix.size()
    proj_matrix = t.mm(matrix, t.transpose(matrix, 1, 0))
    params_vector = parameters_to_vector(model.parameters()).detach()
#	sphere_reg = torch.tensor(0., requires_grad=True)
	params_proj = t.mv(proj_matrix, params_vector)
	offset_proj = t.mv(proj_matrix, offset)
	r_proj_params = t.sqrt(t.norm(params_proj-offset))
	sphere_reg = lambda_sphere*(r_proj_params-radius)**2
	orth_reg = lambda_orth*t.norm(params-offset-params_proj+offset_proj)

    # Combine the initial loss with the weight function
    total_loss = loss + sphere_reg + orth_reg
    return total_loss


def loss_diff(model, input_data, target_data, difference_vector):
    original_loss = loss_fn(model(input_data), target_data)
    params_vector = parameters_to_vector(model.parameters()).detach()
    perturbed_params_vector = params_vector + d 
    #[p + d for p, d in zip(params_vector(), difference_vector)]
    perturbed_params = vector_to_parameters(perturbed_params_vector)
    perturbed_loss = loss_fn(model(input_data, params=perturbed_params), target_data)
    return original_loss - perturbed_loss



def closure():
    optimizer.zero_grad() # Clearing the previous gradients
    output = model(input_data) # Forward pass through the model
    loss = loss_fn(output, target_data) # Computing the loss
    loss.backward() # Computing the gradients
    return loss

def optimizer(self, model, difference_vector):
		pass

def train(data_loader_train, model, criterion, filename, n_epochs, initial_lr, lr_decay, patterns_per_num, print_pure=False):

