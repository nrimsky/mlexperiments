import torch
import numpy as np
from torch.optim.optimizer import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class UniformSampler(object):
    def __init__(self, minval, maxval, cuda=False):
        self.minval = minval
        self.maxval = maxval
        self.cuda = cuda
        self.dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def sample(self, size):
        return self.dtype(*size).uniform_(self.minval, self.maxval)


class GaussianSampler(object):
    def __init__(self, mu, sigma, cuda=False):
        self.sigma = sigma
        self.mu = mu
        self.cuda = cuda
        self.dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def sample(self, size):
        return self.dtype(*size).normal_(self.mu, self.sigma)


class SimulatedAnnealing(Optimizer):
    def __init__(self, params, sampler, tau0=1.0, anneal_rate=0.0003,
                 min_temp=1e-5, anneal_every=100000, hard=False, hard_rate=0.9):
        defaults = dict(sampler=sampler, tau0=tau0, tau=tau0, anneal_rate=anneal_rate,
                        min_temp=min_temp, anneal_every=anneal_every,
                        hard=hard, hard_rate=hard_rate, iteration=0)
        super(SimulatedAnnealing, self).__init__(params, defaults)

    def step(self, closure=None):
        if closure is None:
            raise Exception("loss closure is required to do SA")

        loss = closure()

        params = []
        for group in self.param_groups:
            params.extend(group['params'])

        group = self.param_groups[0]

        if group['iteration'] > 0 and group['iteration'] % group['anneal_every'] == 0:
            if not group['hard']:
                rate = -group['anneal_rate'] * group['iteration']
                group['tau'] = np.maximum(group['tau0'] * np.exp(rate), group['min_temp'])
            else:
                group['tau'] = np.maximum(group['hard_rate'] * group['tau'], group['min_temp'])

        for p in params:
            random_perturbation = group['sampler'].sample(p.data.size())
            p.data = p.data / (torch.norm(p.data) + 1e-8)
            p.data.add_(random_perturbation)

        group['iteration'] += 1

        loss_perturbed = closure()
        final_loss, is_swapped = self.anneal(loss, loss_perturbed, group['tau'])
        if is_swapped:
            vector_to_parameters(parameters_to_vector(params), params)

        return final_loss

    def anneal(self, loss, loss_perturbed, tau):
        def acceptance_prob(old, new, temp):
            return torch.exp((old - new)/temp)

        if loss_perturbed.item() < loss.item():
            return loss_perturbed, True
        else:
            ap = acceptance_prob(loss, loss_perturbed, tau)
            if ap.item() > np.random.rand():
                return loss_perturbed, True
            return loss, False
