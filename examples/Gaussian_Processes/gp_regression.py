"""
GPyTorch Regression Example

Demonstrates how to implement full-batch L-BFGS with weak Wolfe line search
without Powell damping to train an RBF kernel Gaussian process on a simple function
y = sin(2*pi*x) + eps
eps ~ N(0, 0.2).

Based on GPyTorch example; see GPyTorch documentation for more information.

Many thanks to Jacob Gardner, Geoff Pleiss, and Ben Letham for feedback on PyTorch-LBFGS
for training Gaussian processes in GPyTorch.

Implemented by: Hao-Jun Michael Shi
Last edited: 11/17/18.

Requirements:
    - PyTorch
    - GPyTorch (https://gpytorch.ai/)

Run Command:
    python gp_regression.py

"""

import math
import torch
import gpytorch
from matplotlib import pyplot as plt

# added these lines to include PyTorch-LBFGS
import sys
sys.path.append('../../../PyTorch-LBFGS/functions/')
from LBFGS import FullBatchLBFGS

# Training data is 11 points in [0,1] inclusive regularly spaced
train_x = torch.linspace(0, 1, 100)
# True function is sin(2*pi*x) with Gaussian noise
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use full-batch L-BFGS optimizer
optimizer = FullBatchLBFGS(model.parameters())

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# define closure
def closure():
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    return loss

loss = closure()
loss.backward()

training_iter = 10
for i in range(training_iter):

    # perform step and update curvature
    options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
    loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(options)

    print('Iter %d/%d - Loss: %.3f - LR: %.3f - Func Evals: %0.0f - Grad Evals: %0.0f - Log-Lengthscale: %.3f - Log_Noise: %.3f' % (
        i + 1, training_iter, loss.item(), lr, F_eval, G_eval,
        model.covar_module.base_kernel.log_lengthscale.item(),
        model.likelihood.log_noise.item()
        ))

