"""
Multi-Batch L-BFGS Implementation with Fixed Steplength

Demonstrates how to implement multi-batch L-BFGS with fixed steplength and Powell 
damping to train a simple convolutional neural network using the LBFGS optimizer. 
Multi-batch L-BFGS is a stochastic quasi-Newton method that performs curvature 
pair updates over the overlap between consecutive samples in the stochastic gradient.

This implementation is CUDA-compatible.

Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
Last edited 9/21/18.

Requirements:
    - Keras (for CIFAR-10 dataset)
    - NumPy
    - PyTorch

Run Command:
    python multi_batch_lbfgs_example.py

Based on stable quasi-Newton updating introduced by Berahas, Nocedal, and Takac in
"A Multi-Batch L-BFGS Method for Machine Learning" (2016)

"""

import sys
sys.path.append('../../functions/')

import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from keras.datasets import cifar10 # to load dataset

from utils import compute_stats, get_grad
from LBFGS import LBFGS

#%% Parameters for L-BFGS training

max_iter = 200                      # note each iteration is NOT an epoch
ghost_batch = 128
batch_size = 8192
overlap_ratio = 0.25                # should be in (0, 0.5)
lr = 1

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
    torch.cuda.manual_seed(2018)
    model = ConvNet().cuda() 
else:
    torch.manual_seed(2018)
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

optimizer = LBFGS(model.parameters(), lr=lr, history_size=10, line_search='None', debug=True)

#%% Main training loop

Ok_size = int(overlap_ratio*batch_size)
Nk_size = int((1 - 2*overlap_ratio)*batch_size)

# sample previous overlap gradient
random_index = np.random.permutation(range(X_train.shape[0]))
Ok_prev = random_index[0:Ok_size]
g_Ok_prev, obj_Ok_prev = get_grad(optimizer, X_train[Ok_prev], y_train[Ok_prev], opfun)

# main loop
for n_iter in range(max_iter):
    
    # training mode
    model.train()
    
    # sample current non-overlap and next overlap gradient
    random_index = np.random.permutation(range(X_train.shape[0]))
    Ok = random_index[0:Ok_size]
    Nk = random_index[Ok_size:(Ok_size + Nk_size)]
    
    # compute overlap gradient and objective
    g_Ok, obj_Ok = get_grad(optimizer, X_train[Ok], y_train[Ok], opfun)
    
    # compute non-overlap gradient and objective
    g_Nk, obj_Nk = get_grad(optimizer, X_train[Nk], y_train[Nk], opfun)
    
    # compute accumulated gradient over sample
    g_Sk = overlap_ratio*(g_Ok_prev + g_Ok) + (1 - 2*overlap_ratio)*g_Nk
        
    # two-loop recursion to compute search direction
    p = optimizer.two_loop_recursion(-g_Sk)
                
    # perform line search step
    lr = optimizer.step(p, g_Ok, g_Sk=g_Sk)
    
    # compute previous overlap gradient for next sample
    Ok_prev = Ok
    g_Ok_prev, obj_Ok_prev = get_grad(optimizer, X_train[Ok_prev], y_train[Ok_prev], opfun)
    
    # curvature update
    optimizer.curvature_update(g_Ok_prev, eps=0.2, damping=True)
    
    # compute statistics
    model.eval()
    train_loss, test_loss, test_acc = compute_stats(X_train, y_train, X_test, 
                                                    y_test, opfun, accfun, ghost_batch=128)
            
    # print data
    print('Iter:',n_iter+1, 'lr:', lr, 'Training Loss:', train_loss, 
          'Test Loss:', test_loss, 'Test Accuracy:', test_acc)
