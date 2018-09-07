"""
Line Search Comparison on Full-Overlap L-BFGS on CIFAR-10 ConvNet

This implementation is CUDA-compatible.

Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
Last edited 9/6/18.

Requirements:
    - Keras (for CIFAR-10 dataset)
    - NumPy
    - PyTorch

Run Command:
    python line_search_comparison.py

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

max_iter = 1000
ghost_batch = 128
batch_size = 8192

#%% Load data

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train/255
X_test = X_test/255

X_train = np.transpose(X_train, (0, 3, 1, 2))
X_test = np.transpose(X_test, (0, 3, 1, 2))

no_samples = X_train.shape[0]

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

new_instance = ConvNet

#%% Check cuda availability
        
cuda = torch.cuda.is_available()

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

#%% Create neural network model
        
if(cuda):
    model = new_instance().cuda() 
else:
    model = new_instance()
    
#%% Define optimizer

optimizer = LBFGS(model.parameters(), lr=1, history_size=10, line_search='Armijo')

#%% Main training loop

iters_armijo = []
passes_armijo = []
train_loss_armijo = []
test_loss_armijo = []
test_acc_armijo = []
lrs_armijo = []

passes = 0

# main loop
for n_iter in range(max_iter):
    
    # training mode
    model.train()
    
    # sample batch
    random_index = np.random.permutation(range(X_train.shape[0]))
    Sk = random_index[0:batch_size]
    
    # compute initial gradient and objective
    grad, obj = get_grad(optimizer, X_train[Sk], y_train[Sk], opfun)
    passes += 2*batch_size/no_samples
    
    # two-loop recursion to compute search direction
    p = optimizer.two_loop_recursion(-grad)
            
    # define closure for line search
    def closure():              
        
        optimizer.zero_grad()
        
        if(torch.cuda.is_available()):
            loss_fn = torch.tensor(0, dtype=torch.float).cuda()
        else:
            loss_fn = torch.tensor(0, dtype=torch.float)
        
        for subsmpl in np.array_split(Sk, max(int(batch_size/ghost_batch), 1)):
                        
            ops = opfun(X_train[subsmpl])
            
            if(torch.cuda.is_available()):
                tgts = torch.from_numpy(y_train[subsmpl]).cuda().long().squeeze()
            else:
                tgts = torch.from_numpy(y_train[subsmpl]).long().squeeze()
                
            loss_fn += F.cross_entropy(ops, tgts)*(len(subsmpl)/batch_size)
                        
        return loss_fn
    
    # perform line search step
    options = {'closure': closure, 'current_loss': obj, 'eta': 2}
    obj, lr, _, clos_evals, flag = optimizer.step(p, grad, options)
    passes += clos_evals*batch_size/no_samples
    
    # compute gradient at new iterate
    obj.backward()
    grad = optimizer._gather_flat_grad()
    passes += batch_size/no_samples
    
    # curvature update
    optimizer.curvature_update(grad)
    
    # compute statistics
    model.eval()
    train_loss, test_loss, test_acc = compute_stats(X_train, y_train, X_test, 
                                                    y_test, opfun, accfun, ghost_batch=128)
            
    # print data
    print('Iter:',n_iter+1, 'lr:', lr, 'Training Loss:', train_loss, 
          'Test Loss:', test_loss, 'Test Accuracy:', test_acc)
    
    # store data
    iters_armijo.append(n_iter+1)
    passes_armijo.append(passes)
    train_loss_armijo.append(train_loss)
    test_loss_armijo.append(test_loss)
    test_acc_armijo.append(test_acc)
    lrs_armijo.append(lr)
    
np.savez('CIFAR10_ConvNet' + '_' + 'LBFGS-FO' + '_' + 'Armijo' + '_' + str(max_iter) + '_' + str(batch_size) + '_' + str(ghost_batch),
         iters=iters_armijo, passes=passes_armijo, train_loss=train_loss_armijo, test_loss=test_loss_armijo, test_acc=test_acc_armijo,
        lrs=lrs_armijo)

#%% Create neural network model
        
if(cuda):
    model = new_instance().cuda() 
else:
    model = new_instance()
    
#%% Define optimizer

optimizer = LBFGS(model.parameters(), lr=1, history_size=10, line_search='Armijo')

#%% Main training loop

iters_armijoi = []
passes_armijoi = []
train_loss_armijoi = []
test_loss_armijoi = []
test_acc_armijoi = []
lrs_armijoi = []

passes = 0

# main loop
for n_iter in range(max_iter):
    
    # training mode
    model.train()
    
    # sample batch
    random_index = np.random.permutation(range(X_train.shape[0]))
    Sk = random_index[0:batch_size]
    
    # compute initial gradient and objective
    grad, obj = get_grad(optimizer, X_train[Sk], y_train[Sk], opfun)
    passes += 2*batch_size/no_samples
    
    # two-loop recursion to compute search direction
    p = optimizer.two_loop_recursion(-grad)
            
    # define closure for line search
    def closure():              
        
        optimizer.zero_grad()
        
        if(torch.cuda.is_available()):
            loss_fn = torch.tensor(0, dtype=torch.float).cuda()
        else:
            loss_fn = torch.tensor(0, dtype=torch.float)
        
        for subsmpl in np.array_split(Sk, max(int(batch_size/ghost_batch), 1)):
                        
            ops = opfun(X_train[subsmpl])
            
            if(torch.cuda.is_available()):
                tgts = torch.from_numpy(y_train[subsmpl]).cuda().long().squeeze()
            else:
                tgts = torch.from_numpy(y_train[subsmpl]).long().squeeze()
                
            loss_fn += F.cross_entropy(ops, tgts)*(len(subsmpl)/batch_size)
                        
        return loss_fn
    
    # perform line search step
    options = {'closure': closure, 'current_loss': obj, 'eta': 2}
    obj, lr, _, clos_evals, flag = optimizer.step(p, grad, options)
    passes += clos_evals*batch_size/no_samples
    
    # compute gradient at new iterate
    obj.backward()
    grad = optimizer._gather_flat_grad()
    passes += batch_size/no_samples
    
    # curvature update
    optimizer.curvature_update(grad)
    
    # compute statistics
    model.eval()
    train_loss, test_loss, test_acc = compute_stats(X_train, y_train, X_test, 
                                                    y_test, opfun, accfun, ghost_batch=128)
            
    # print data
    print('Iter:',n_iter+1, 'lr:', lr, 'Training Loss:', train_loss, 
          'Test Loss:', test_loss, 'Test Accuracy:', test_acc)
    
    # store data
    iters_armijoi.append(n_iter+1)
    passes_armijoi.append(passes)
    train_loss_armijoi.append(train_loss)
    test_loss_armijoi.append(test_loss)
    test_acc_armijoi.append(test_acc)
    lrs_armijoi.append(lr)

np.savez('CIFAR10_ConvNet' + '_' + 'LBFGS-FO' + '_' + 'Armijo_Interp' + '_' + str(max_iter) + '_' + str(batch_size) + '_' + str(ghost_batch),
         iters=iters_armijoi, passes=passes_armijoi, train_loss=train_loss_armijoi, test_loss=test_loss_armijoi, test_acc=test_acc_armijoi,
        lrs=lrs_armijoi)
    
#%% Create neural network model
        
if(cuda):
    model = new_instance().cuda() 
else:
    model = new_instance()
    
#%% Define optimizer

optimizer = LBFGS(model.parameters(), lr=1, history_size=10, line_search='Wolfe')

#%% Main training loop

iters_wolfe = []
passes_wolfe = []
train_loss_wolfe = []
test_loss_wolfe = []
test_acc_wolfe = []
lrs_wolfe = []

passes = 0

# main loop
for n_iter in range(max_iter):
    
    # training mode
    model.train()
    
    # sample batch
    random_index = np.random.permutation(range(X_train.shape[0]))
    Sk = random_index[0:batch_size]
    
    # compute initial gradient and objective
    grad, obj = get_grad(optimizer, X_train[Sk], y_train[Sk], opfun)
    passes += 2*batch_size/no_samples
    
    # two-loop recursion to compute search direction
    p = optimizer.two_loop_recursion(-grad)
            
    # define closure for line search
    def closure():              
        
        optimizer.zero_grad()
        
        if(torch.cuda.is_available()):
            loss_fn = torch.tensor(0, dtype=torch.float).cuda()
        else:
            loss_fn = torch.tensor(0, dtype=torch.float)
        
        for subsmpl in np.array_split(Sk, max(int(batch_size/ghost_batch), 1)):
                        
            ops = opfun(X_train[subsmpl])
            
            if(torch.cuda.is_available()):
                tgts = torch.from_numpy(y_train[subsmpl]).cuda().long().squeeze()
            else:
                tgts = torch.from_numpy(y_train[subsmpl]).long().squeeze()
                
            loss_fn += F.cross_entropy(ops, tgts)*(len(subsmpl)/batch_size)
                        
        return loss_fn
    
    # perform line search step
    options = {'closure': closure, 'current_loss': obj, 'eta': 2}
    obj, grad, lr, _, clos_evals, grad_evals, flag = optimizer.step(p, grad, options)
    passes += (clos_evals+grad_evals)*batch_size/no_samples
    
    # curvature update
    optimizer.curvature_update(grad)
    
    # compute statistics
    model.eval()
    train_loss, test_loss, test_acc = compute_stats(X_train, y_train, X_test, 
                                                    y_test, opfun, accfun, ghost_batch=128)
            
    # print data
    print('Iter:',n_iter+1, 'lr:', lr, 'Training Loss:', train_loss, 
          'Test Loss:', test_loss, 'Test Accuracy:', test_acc)
    
    # store data
    iters_wolfe.append(n_iter+1)
    passes_wolfe.append(passes)
    train_loss_wolfe.append(train_loss)
    test_loss_wolfe.append(test_loss)
    test_acc_wolfe.append(test_acc)
    lrs_wolfe.append(lr)

np.savez('CIFAR10_ConvNet' + '_' + 'LBFGS-FO' + '_' + 'Wolfe' + '_' + str(max_iter) + '_' + str(batch_size) + '_' + str(ghost_batch),
         iters=iters_wolfe, passes=passes_wolfe, train_loss=train_loss_wolfe, test_loss=test_loss_wolfe, test_acc=test_acc_wolfe,
        lrs=lrs_wolfe)
