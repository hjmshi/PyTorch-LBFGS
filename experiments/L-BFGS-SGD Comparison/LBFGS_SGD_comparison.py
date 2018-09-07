"""
Comparison of L-BFGS with backtracking line search and SGD

Implemented by: Hao-Jun Michael Shi
Last edited 8/24/18.

Requirements:
    - Keras (for datasets)
    - NumPy
    - PyTorch

Run Command:
    python LBFGS_SGD_Comparison.py

"""

import sys
sys.path.append('../../functions/')

import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from keras.datasets import cifar10, mnist # to load dataset

from utils import compute_stats, get_grad
from LBFGS import LBFGS

#%% Parameters

# Main Parameters
problem_name = 'MNIST_LogisticRegression'
epochs = 50
ghost_batch = 128
batch_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
seed = 1226

# SGD Parameters
lr_sgd = 10
momentum = 0.9

# L-BFGS Parameters
lr_lbfgs = 0.1
overlap_ratio = 0.375
line_search = 'Armijo'
history_size = 100

#%% Load MNIST data

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32').reshape((60000, 784))
X_test = X_test.astype('float32').reshape((10000, 784))
X_train /= 255
X_test /= 255

#%% Load CIFAR-10 data

#(X_train, y_train), (X_test, y_test) = cifar10.load_data()
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
#X_train = X_train/255
#X_test = X_test/255
#
#X_train = np.transpose(X_train, (0, 3, 1, 2))
#X_test = np.transpose(X_test, (0, 3, 1, 2))


no_samples = X_train.shape[0]

#%% Define Logistic Regression
        
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.classifier = nn.Sequential(
                nn.Linear(784, 10)
                )
        
    def forward(self, x):
        out = self.classifier(x)
        return out
    
new_instance = lambda: LogisticRegression()

#%% Define Convolutional Network

#class ConvNet(nn.Module):
#    def __init__(self):
#        super(ConvNet, self).__init__()
#        self.conv1 = nn.Conv2d(3, 6, 5)
#        self.pool = nn.MaxPool2d(2, 2)
#        self.conv2 = nn.Conv2d(6, 16, 5)
#        self.fc1 = nn.Linear(16 * 5 * 5, 1000)
#        self.fc2 = nn.Linear(1000, 10)
#        
#    def forward(self, x):
#        x = self.pool(F.relu(self.conv1(x)))
#        x = self.pool(F.relu(self.conv2(x)))
#        x = x.view(-1, 16 * 5 * 5)
#        x = F.relu(self.fc1(x))
#        x = self.fc2(x)
#        return x
#
#new_instance = lambda: ConvNet()

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

#%% Compute norm of gradient

def compute_grad_norm(parameters):
    
    norm = 0
    
    for param in parameters:
        if(param is not None):
            if(cuda):
                norm += np.linalg.norm(param.grad.cpu().numpy())**2
            else:
                norm += np.linalg.norm(param.grad.numpy())**2
            
    norm = np.sqrt(norm)
        
    return norm

#%% Iterate over all batch sizes
for batch_size in batch_sizes:
    
    #%% SGD
    print('SGD', batch_size)
    
    # initialize data
    iters_sgd = []
    epochs_sgd = [] # amount of data touched
    train_loss_sgd = []
    test_loss_sgd = []
    test_acc_sgd = []
    grad_norm_sgd = []
    lrs_sgd = []
    passes_sgd = [] # number of forward or backward passes
    
    np.random.seed(seed)
                
    if(cuda):
        torch.cuda.manual_seed(seed)
        model = new_instance().cuda()
    else:
        torch.manual_seed(seed)
        model = new_instance()
        
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_sgd, momentum=momentum)
    
    n_iter = 0
    e = 0
    passes = 0
    
    # main loop
    while e < epochs:
        
        for Sk in np.array_split(np.random.permutation(range(no_samples)), np.ceil(no_samples/batch_size)):
        
            # training mode
            model.train()
            
            # sample batch
            random_index = np.random.permutation(range(no_samples))
            Sk = random_index[0:batch_size]
            
            # zero gradient
            optimizer.zero_grad()
            
            # processing data in chunks
            for subsmpl in np.array_split(Sk, int(batch_size/ghost_batch)):
                
                # compute forward pass
                ops = opfun(X_train[subsmpl])
                
                if(torch.cuda.is_available()):
                    tgts = torch.from_numpy(y_train[subsmpl]).cuda().long().squeeze()
                else:
                    tgts = torch.from_numpy(y_train[subsmpl]).long().squeeze()
    
                loss_fn = F.cross_entropy(ops, tgts)*(len(subsmpl)/batch_size)
                
                # compute backward pass
                loss_fn.backward()
            
            # take optimization step
            optimizer.step()
            
            # iterate
            n_iter += 1
            e += len(Sk)/no_samples
            passes += 2*len(Sk)/no_samples
            
            # compute statistics
            model.eval()
            train_loss, test_loss, test_acc = compute_stats(X_train, y_train, X_test, 
                                                            y_test, opfun, accfun, ghost_batch=128)
        
            # compute norm of gradient
            optimizer.zero_grad()
        
            # processing data in chunks
            for subsmpl in np.array_split(np.arange(no_samples), max(int(no_samples/ghost_batch), 1)):
            
                # compute forward pass
                ops = opfun(X_train[subsmpl])
            
                if(torch.cuda.is_available()):
                    tgts = torch.from_numpy(y_train[subsmpl]).cuda().long().squeeze()
                else:
                    tgts = torch.from_numpy(y_train[subsmpl]).long().squeeze()
    
                loss_fn = F.cross_entropy(ops, tgts)*(len(subsmpl)/no_samples)
        
                # compute backward pass
                loss_fn.backward()
            
            grad_norm = compute_grad_norm(model.parameters())
        
            # append data
            iters_sgd.append(n_iter)
            epochs_sgd.append(e)
            train_loss_sgd.append(train_loss)
            test_loss_sgd.append(test_loss)
            test_acc_sgd.append(test_acc)
            grad_norm_sgd.append(grad_norm)
            lrs_sgd.append(optimizer.param_groups[0]['lr'])
            passes_sgd.append(passes)
        
            # print data
            print('Epoch:', e, 'Props:', passes, 'Iter:',n_iter, 'LR:', 
                  optimizer.param_groups[0]['lr'], 'Training Loss:', train_loss, 
                  'Gradient Norm:', grad_norm, 'Test Loss:', test_loss, 
                  'Test Accuracy:', test_acc)
                                        
    # store data
    file_name = problem_name + '_SGD_' + str(epochs) + '_' + str(batch_size) + '_' + str(lr_sgd) + '_' + str(momentum) + '_' + str(seed)
    np.savez(file_name, iters=iters_sgd, epochs=epochs_sgd, train_loss=train_loss_sgd, 
             test_loss=test_loss_sgd, test_acc=test_acc_sgd, grad_norm=grad_norm_sgd,
             lr=lrs_sgd, passes=passes_sgd)
        
    #%% Multi-Batch L-BFGS
    print('L-BFGS-MB', batch_size)
    
    # initialize data
    iters_lbfgs_mb = []
    epochs_lbfgs_mb = [] # amount of data touched
    train_loss_lbfgs_mb = []
    test_loss_lbfgs_mb = []
    test_acc_lbfgs_mb = []
    grad_norm_lbfgs_mb = []
    lrs_lbfgs_mb = []
    passes_lbfgs_mb = [] # number of forward or backward passes
    backtracks_lbfgs_mb = [] # number of backtracks in line search
    
    # initialize model
    np.random.seed(seed)
    
    if(cuda):
        torch.cuda.manual_seed(seed)
        model = new_instance().cuda() 
    else:
        torch.manual_seed(seed)
        model = new_instance()
        
    # track stats
    n_iter = 0
    e = 0
    passes = 0
    
    # define optimizer
    optimizer = LBFGS(model.parameters(), lr=lr_lbfgs, history_size=history_size, line_search=line_search)
    
    # define sample sizes
    Ok_size = int(overlap_ratio*batch_size)
    Nk_size = int((1 - 2*overlap_ratio)*batch_size)
    
    # sample previous overlap gradient
    random_index = np.random.permutation(range(no_samples))
    Ok_prev = random_index[0:Ok_size]
    g_Ok_prev, obj_Ok_prev = get_grad(optimizer, X_train[Ok_prev], y_train[Ok_prev], opfun)
    passes += 2*Ok_size/no_samples
        
    # main loop
    while e < epochs:
        
        # training mode
        model.train()
        
        # sample current non-overlap and next overlap gradient
        random_index = np.random.permutation(range(no_samples))
        Ok = random_index[0:Ok_size]
        Nk = random_index[Ok_size:(Ok_size + Nk_size)]
        
        # compute overlap gradient and objective
        g_Ok, obj_Ok = get_grad(optimizer, X_train[Ok], y_train[Ok], opfun)
        passes += 2*Ok_size/no_samples
        
        # compute non-overlap gradient and objective
        g_Nk, obj_Nk = get_grad(optimizer, X_train[Nk], y_train[Nk], opfun)
        passes += 2*Nk_size/no_samples
        
        # compute accumulated gradient and objective over sample
        g_Sk = overlap_ratio*(g_Ok_prev + g_Ok) + (1 - 2*overlap_ratio)*g_Nk
        
        # concatenate to obtain Sk = {O_{k - 1}, N_k, O_k}
        Sk = np.concatenate((Ok_prev, Nk, Ok), axis=0)
        
        # two-loop recursion to compute search direction
        p = optimizer.two_loop_recursion(-g_Sk)
                
        # define closure for line search over overlap
        def closure():
            
            optimizer.zero_grad()
            if(torch.cuda.is_available()):
                loss_fn = torch.tensor(0, dtype=torch.float).cuda()
            else:
                loss_fn = torch.tensor(0, dtype=torch.float)
                
            for subsmpl in np.array_split(Ok, max(int(Ok_size/ghost_batch), 1)):
                
                ops = opfun(X_train[subsmpl])
                
                if(torch.cuda.is_available()):
                    tgts = torch.from_numpy(y_train[subsmpl]).cuda().long().squeeze()
                else:
                    tgts = torch.from_numpy(y_train[subsmpl]).long().squeeze()
                    
                loss_fn += F.cross_entropy(ops, tgts)*(len(subsmpl)/Ok_size)
                                
            return loss_fn
            
        # perform line search step over Ok sample
        options = {'closure': closure, 'current_loss': obj_Ok, 'eta': 2}
        obj_Ok_prev, lr, backtracks, clos_eval = optimizer.step(p, g_Ok, options)
        passes += clos_eval*Ok_size/no_samples
        
        # compute overlap gradient at new iterate
        Ok_prev = Ok
        obj_Ok_prev.backward()
        g_Ok_prev = optimizer._gather_flat_grad()
        passes += Ok_size/no_samples
        
        # curvature update
        optimizer.curvature_update(g_Ok_prev)
        
        # iterate
        n_iter += 1
        e += (Ok_size + Nk_size)/no_samples
        
        # compute statistics
#        if(n_iter % int(no_samples/batch_size) == 0):
        if(True):
            model.eval()
            train_loss, test_loss, test_acc = compute_stats(X_train, y_train, X_test, 
                                                            y_test, opfun, accfun, ghost_batch=128)
            
            # compute norm of gradient
            optimizer.zero_grad()
            
            # processing data in chunks
            for subsmpl in np.array_split(np.arange(no_samples), max(int(no_samples/ghost_batch), 1)):
                
                # compute forward pass
                ops = opfun(X_train[subsmpl])
                
                if(torch.cuda.is_available()):
                    tgts = torch.from_numpy(y_train[subsmpl]).cuda().long().squeeze()
                else:
                    tgts = torch.from_numpy(y_train[subsmpl]).long().squeeze()
    
                loss_fn = F.cross_entropy(ops, tgts)*(len(subsmpl)/no_samples)
            
                # compute backward pass
                loss_fn.backward()
                
            grad_norm = compute_grad_norm(model.parameters())
            
            # append data
            iters_lbfgs_mb.append(n_iter)
            epochs_lbfgs_mb.append(e)
            train_loss_lbfgs_mb.append(train_loss)
            test_loss_lbfgs_mb.append(test_loss)
            test_acc_lbfgs_mb.append(test_acc)
            grad_norm_lbfgs_mb.append(grad_norm)
            lrs_lbfgs_mb.append(lr)
            passes_lbfgs_mb.append(passes)
            backtracks_lbfgs_mb.append(backtracks)
            
            # print data
            print('Epoch:', e, 'Props:', passes, 'Iter:',n_iter, 'LR:', lr,
                  'Training Loss:', train_loss, 'Gradient Norm:', grad_norm, 
                  'Test Loss:', test_loss, 'Test Accuracy:', test_acc)
        
    # store data
    file_name = problem_name + '_L-BFGS-MB_' + str(epochs) + '_' + str(batch_size) + '_' + \
    str(lr_lbfgs) + '_' + str(history_size) + '_' + str(overlap_ratio) + '_' + \
    line_search + '_' + str(seed)
    np.savez(file_name, iters=iters_lbfgs_mb, epochs=epochs_lbfgs_mb, train_loss=train_loss_lbfgs_mb, 
             test_loss=test_loss_lbfgs_mb, test_acc=test_acc_lbfgs_mb, grad_norm=grad_norm_lbfgs_mb,
             lr=lrs_lbfgs_mb, passes=passes_lbfgs_mb, backtracks=backtracks_lbfgs_mb)
        
    #%% Full-Overlap L-BFGS
    print('L-BFGS-FO', batch_size)
        
    # initialize data
    iters_lbfgs_fo = []
    epochs_lbfgs_fo = []
    train_loss_lbfgs_fo = []
    test_loss_lbfgs_fo = []
    test_acc_lbfgs_fo = []
    grad_norm_lbfgs_fo = []
    lrs_lbfgs_fo = []
    passes_lbfgs_fo = [] # number of forward or backward passes
    backtracks_lbfgs_fo = [] # number of backtracks in line search
    
    # initialize model
    np.random.seed(seed)
    
    if(cuda):
        torch.cuda.manual_seed(seed)
        model = new_instance().cuda() 
    else:
        torch.manual_seed(seed)
        model = new_instance()
    
    # track stats
    n_iter = 0
    e = 0
    passes = 0
    
    # define optimizer
    optimizer = LBFGS(model.parameters(), lr=lr_lbfgs, history_size=history_size, line_search=line_search)    
        
    # main loop
    while e < epochs:
        
        # training mode
        model.train()
        
        # sample batch
        random_index = np.random.permutation(range(no_samples))
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
            
            for subsmpl in np.array_split(Sk, int(batch_size/ghost_batch)):
                            
                ops = opfun(X_train[subsmpl])
                
                if(torch.cuda.is_available()):
                    tgts = torch.from_numpy(y_train[subsmpl]).cuda().long().squeeze()
                else:
                    tgts = torch.from_numpy(y_train[subsmpl]).long().squeeze()
                    
                loss_fn += F.cross_entropy(ops, tgts)*(len(subsmpl)/batch_size)
                                
            return loss_fn
        
        # perform line search step
        options = {'closure': closure, 'current_loss': obj, 'eta': 2}
        obj, lr, backtracks, clos_eval = optimizer.step(p, grad, options)
        passes += clos_eval*batch_size/no_samples
        
        # compute gradient at new iterate
        obj.backward()
        grad = optimizer._gather_flat_grad()
        passes += batch_size/no_samples
        
        # curvature update
        optimizer.curvature_update(grad)
        
        # iterate
        n_iter += 1
        e += batch_size/no_samples
        
        # compute statistics
#        if(n_iter % int(no_samples/batch_size) == 0):
        if(True):
            model.eval()
            train_loss, test_loss, test_acc = compute_stats(X_train, y_train, X_test, 
                                                            y_test, opfun, accfun, ghost_batch=128)
            
            # compute norm of gradient
            optimizer.zero_grad()
            
            # processing data in chunks
            for subsmpl in np.array_split(np.arange(no_samples), max(int(no_samples/ghost_batch), 1)):
                
                # compute forward pass
                ops = opfun(X_train[subsmpl])
                
                if(torch.cuda.is_available()):
                    tgts = torch.from_numpy(y_train[subsmpl]).cuda().long().squeeze()
                else:
                    tgts = torch.from_numpy(y_train[subsmpl]).long().squeeze()
    
                loss_fn = F.cross_entropy(ops, tgts)*(len(subsmpl)/no_samples)
            
                # compute backward pass
                loss_fn.backward()
                
            grad_norm = compute_grad_norm(model.parameters())
                
            # append data
            iters_lbfgs_fo.append(n_iter)
            epochs_lbfgs_fo.append(e)
            train_loss_lbfgs_fo.append(train_loss)
            test_loss_lbfgs_fo.append(test_loss)
            test_acc_lbfgs_fo.append(test_acc)
            grad_norm_lbfgs_fo.append(grad_norm)
            lrs_lbfgs_fo.append(lr)
            passes_lbfgs_fo.append(passes)
            backtracks_lbfgs_fo.append(backtracks)
    
            # print data
            print('Epoch:', e, 'Props:', passes, 'Iter:',n_iter, 'LR:', lr,
                  'Training Loss:', train_loss, 'Gradient Norm:', grad_norm, 
                  'Test Loss:', test_loss, 'Test Accuracy:', test_acc)
        
    # store data
    file_name = problem_name + '_L-BFGS-FO_' + str(epochs) + '_' + str(batch_size) + '_' + \
    str(lr_lbfgs) + '_' + str(history_size) + '_' + \
    line_search + '_' + str(seed)
    np.savez(file_name, iters=iters_lbfgs_fo, epochs=epochs_lbfgs_fo, train_loss=train_loss_lbfgs_fo, 
             test_loss=test_loss_lbfgs_fo, test_acc=test_acc_lbfgs_fo, grad_norm=grad_norm_lbfgs_fo,
             lr=lrs_lbfgs_fo, passes=passes_lbfgs_fo, backtracks=backtracks_lbfgs_fo)
