"""
Full-Batch L-BFGS Implementation

Demonstrates how to implement a simple full-batch L-BFGS with Armijo backtracking 
line search to train a simple convolutional neural network using the LBFGS optimizer.

This implementation is CUDA-compatible.

Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
Last edited 9/6/18.

Requirements:
    - Keras (for CIFAR-10 dataset)
    - NumPy
    - PyTorch

Run Command:
    python full_batch_lbfgs_example.py

"""

import sys
sys.path.append('../functions/')

import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from keras.datasets import cifar10 # to load dataset

from utils import compute_stats, get_grad
from LBFGS import LBFGS

#%% Parameters for L-BFGS training

max_iter = 100
ghost_batch = 32

#%% Load data

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train/255
X_test = X_test/255

X_train = np.transpose(X_train, (0, 3, 1, 2))
X_test = np.transpose(X_test, (0, 3, 1, 2))

#%% Define network

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 1000)
        self.fc2 = nn.Linear(1000, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#%% Check cuda availability
        
cuda = torch.cuda.is_available()
    
#%% Create neural network model
        
if(cuda):
    model = ConvNet().cuda() 
else:
    model = ConvNet()
    
#%% Define helper functions

# Forward pass
if(cuda):
    opfun = lambda X: model.forward(torch.from_numpy(X).cuda())
else:
    opfun = lambda X: model.forward(torch.from_numpy(X))

# Forward pass through the network given the input
if(cuda):
    predsfun = lambda op: np.argmax(op.cpu().data.numpy(), 1)
else:
    predsfun = lambda op: np.argmax(op.data.numpy(), 1)

# Do the forward pass, then compute the accuracy
accfun   = lambda op, y: np.mean(np.equal(predsfun(op), y.squeeze()))*100

#%% Define optimizer

optimizer = LBFGS(model.parameters(), lr=1, history_size=10, line_search='Armijo')

#%% Main training loop

no_samples = X_train.shape[0]

# compute initial gradient and objective
grad, obj = get_grad(optimizer, X_train, y_train, opfun)

# main loop
for n_iter in range(max_iter):
                
    # training mode
    model.train()
    
    # two-loop recursion to compute search direction
    p = optimizer.two_loop_recursion(-grad)
            
    # define closure for line search
    def closure():              
             
        optimizer.zero_grad()
        
        if(torch.cuda.is_available()):
            loss_fn = torch.tensor(0, dtype=torch.float).cuda()
        else:
            loss_fn = torch.tensor(0, dtype=torch.float)
        
        for subsmpl in np.array_split(np.arange(no_samples), max(int(no_samples/ghost_batch), 1)):
                        
            ops = opfun(X_train[subsmpl])
            
            if(torch.cuda.is_available()):
                tgts = torch.from_numpy(y_train[subsmpl]).cuda().long().squeeze()
            else:
                tgts = torch.from_numpy(y_train[subsmpl]).long().squeeze()
                
            loss_fn += F.cross_entropy(ops, tgts)*(len(subsmpl)/no_samples)
                                    
        return loss_fn
                
    # perform line search step
    options = {'closure': closure, 'current_loss': obj, 'eta': 2}
    obj, lr, _, _, _ = optimizer.step(p, grad, options)
    
    # compute gradient at new iterate
    obj.backward()
    grad = optimizer._gather_flat_grad()
    
    # curvature update
    optimizer.curvature_update(grad)
    
    # compute statistics
    model.eval()
    train_loss, test_loss, test_acc = compute_stats(X_train, y_train, X_test, 
                                                    y_test, opfun, accfun, ghost_batch=128)
            
    # print data
    print('Iter:',n_iter+1, 'lr:', lr, 'Training Loss:', train_loss, 
          'Test Loss:', test_loss, 'Test Accuracy:', test_acc)
