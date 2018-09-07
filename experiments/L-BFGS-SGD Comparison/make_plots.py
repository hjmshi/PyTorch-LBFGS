"""
Plot Results from Comparison
"""

import os
import numpy as np
import matplotlib.pyplot as plt

#%% Select Parameters

# Main Parameters
problem_name = 'MNIST_LogisticRegression'
extra = ''
epochs = 50
ghost_batch = 128
seed = 1226
save = True

no_samples = 60000

# SGD Parameters
batch_size_sgd = 8192
lr_sgd = 1
momentum = 0

# L-BFGS-MB Parameters
batch_size_lbfgs_mb = 8192
lr_lbfgs_mb = 1
overlap_ratio = 0.25
line_search_lbfgs_mb = 'Armijo'
history_size_lbfgs_mb = 10

# L-BFGS-FO Parameters
batch_size_lbfgs_fo = 8192
lr_lbfgs_fo = 1
line_search_lbfgs_fo = 'Armijo'
history_size_lbfgs_fo = 10

# Parameters for Plots
loss_lim = [0, 1]
acc_lim = [80, 95]
passes_lim = [0, 10]
iter_lim = [0, 350]
gap = 1 # plot every gap number of iterations

#%% Load Saved Data

os.chdir('./Data/')

file_name_sgd = problem_name + '_SGD_' + str(epochs) + '_' + str(batch_size_sgd) + \
'_' + str(lr_sgd) + '_' + str(momentum) + '_' + str(seed) + '.npz'

file_name_lbfgs_mb = problem_name + '_L-BFGS-MB_' + str(epochs) + '_' + str(batch_size_lbfgs_mb) + '_' + \
str(lr_lbfgs_mb) + '_' + str(history_size_lbfgs_mb) + '_' + str(overlap_ratio) + '_' + \
line_search_lbfgs_mb + '_' + str(seed) + '.npz'

file_name_lbfgs_fo = problem_name + '_L-BFGS-FO_' + str(epochs) + '_' + str(batch_size_lbfgs_fo) + '_' + \
str(lr_lbfgs_fo) + '_' + str(history_size_lbfgs_fo) + '_' + \
line_search_lbfgs_fo + '_' + str(seed) + '.npz'

# Load SGD Data
data_sgd = np.load(file_name_sgd)
epochs_sgd = data_sgd['epochs'][::gap]
iters_sgd = data_sgd['iters'][::gap]
train_loss_sgd = data_sgd['train_loss'][::gap]
test_loss_sgd = data_sgd['test_loss'][::gap]
test_acc_sgd = data_sgd['test_acc'][::gap]
grad_norm_sgd = data_sgd['grad_norm'][::gap]*(batch_size_sgd/no_samples)
lrs_sgd = data_sgd['lr'][::gap]
passes_sgd = data_sgd['passes'][::gap]

# Load L-BFGS-MB Data
data_lbfgs_mb = np.load(file_name_lbfgs_mb)
epochs_lbfgs_mb = data_lbfgs_mb['epochs'][::gap]
iters_lbfgs_mb = data_lbfgs_mb['iters'][::gap]
train_loss_lbfgs_mb = data_lbfgs_mb['train_loss'][::gap]
test_loss_lbfgs_mb = data_lbfgs_mb['test_loss'][::gap]
test_acc_lbfgs_mb = data_lbfgs_mb['test_acc'][::gap]
grad_norm_lbfgs_mb = data_lbfgs_mb['grad_norm'][::gap]*(batch_size_lbfgs_mb/no_samples)
lrs_lbfgs_mb = data_lbfgs_mb['lr'][::gap]
passes_lbfgs_mb = data_lbfgs_mb['passes'][::gap]
backtracks_lbfgs_mb = data_lbfgs_mb['backtracks'][::gap]

# Load L-BFGS-FO Data
data_lbfgs_fo = np.load(file_name_lbfgs_fo)
epochs_lbfgs_fo = data_lbfgs_fo['epochs'][::gap]
iters_lbfgs_fo = data_lbfgs_fo['iters'][::gap]
train_loss_lbfgs_fo = data_lbfgs_fo['train_loss'][::gap]
test_loss_lbfgs_fo = data_lbfgs_fo['test_loss'][::gap]
test_acc_lbfgs_fo = data_lbfgs_fo['test_acc'][::gap]
grad_norm_lbfgs_fo = data_lbfgs_fo['grad_norm'][::gap]*(batch_size_lbfgs_fo/no_samples)
lrs_lbfgs_fo = data_lbfgs_fo['lr'][::gap]
passes_lbfgs_fo = data_lbfgs_fo['passes'][::gap]
backtracks_lbfgs_fo = data_lbfgs_fo['backtracks'][::gap]

#%% Make Plots

os.chdir('../Graphs/')

leg = ['SGD', 'LBFGS-MB', 'LBFGS-FO']
label = (extra + str(batch_size_sgd) + '_' + str(lr_sgd) + '_' + str(momentum) + '_' 
+ str(batch_size_lbfgs_mb) + '_' + str(lr_lbfgs_mb) + '_' + str(overlap_ratio) + '_' + line_search_lbfgs_mb + '_' + str(history_size_lbfgs_mb) + '_'
+ str(batch_size_lbfgs_fo) + '_' + str(lr_lbfgs_fo) + '_' + line_search_lbfgs_fo + '_' + str(history_size_lbfgs_fo) + '.pdf')

# plot passes vs training loss
fig1 = plt.figure()
plt.plot(passes_sgd, train_loss_sgd, passes_lbfgs_mb, train_loss_lbfgs_mb,
         passes_lbfgs_fo, train_loss_lbfgs_fo)
plt.xlabel('Forward/Backward Passes')
plt.ylabel('Train Loss')
plt.xlim(passes_lim)
plt.ylim(loss_lim)
plt.title(problem_name)
plt.legend(leg)
if(save):
    plt.savefig(problem_name + '_passes_train_loss_' + label)

# plot passes vs gradient norm
fig2 = plt.figure()
plt.plot(passes_sgd, grad_norm_sgd, passes_lbfgs_mb, grad_norm_lbfgs_mb,
         passes_lbfgs_fo, grad_norm_lbfgs_fo)
plt.xlabel('Forward/Backward Passes')
plt.ylabel('Gradient Norm')
plt.xlim(passes_lim)
plt.title(problem_name)
plt.legend(leg)
if(save):
    plt.savefig(problem_name + '_passes_grad_norm_' + label)

# plot passes vs test loss
fig3 = plt.figure()
plt.plot(passes_sgd, test_loss_sgd, passes_lbfgs_mb, test_loss_lbfgs_mb,
         passes_lbfgs_fo, test_loss_lbfgs_fo)
plt.xlabel('Forward/Backward Passes')
plt.ylabel('Test Loss')
plt.xlim(passes_lim)
plt.ylim(loss_lim)
plt.title(problem_name)
plt.legend(leg)
if(save):
    plt.savefig(problem_name + '_passes_test_loss_' + label)

# plot passes vs test accuracy
fig4 = plt.figure()
plt.plot(passes_sgd, test_acc_sgd, passes_lbfgs_mb, test_acc_lbfgs_mb,
         passes_lbfgs_fo, test_acc_lbfgs_fo)
plt.xlabel('Forward/Backward Passes')
plt.ylabel('Test Accuracy')
plt.xlim(passes_lim)
plt.ylim(acc_lim)
plt.title(problem_name)
plt.legend(leg)
if(save):
    plt.savefig(problem_name + '_passes_test_acc_' + label)

# plot iters vs training loss
fig5 = plt.figure()
plt.plot(iters_sgd, train_loss_sgd, iters_lbfgs_mb, train_loss_lbfgs_mb,
         iters_lbfgs_fo, train_loss_lbfgs_fo)
plt.xlabel('Iterations')
plt.ylabel('Train Loss')
plt.xlim(iter_lim)
plt.ylim(loss_lim)
plt.title(problem_name)
plt.legend(leg)
if(save):
    plt.savefig(problem_name + '_iters_train_loss_' + label)

# plot iters vs gradient norm
fig6 = plt.figure()
plt.plot(iters_sgd, grad_norm_sgd, iters_lbfgs_mb, grad_norm_lbfgs_mb,
         iters_lbfgs_fo, grad_norm_lbfgs_fo)
plt.xlabel('Iterations')
plt.ylabel('Gradient Norm')
plt.xlim(iter_lim)
plt.title(problem_name)
plt.legend(leg)
if(save):
    plt.savefig(problem_name + '_iters_grad_norm_' + label)

# plot iters vs test loss
fig7 = plt.figure()
plt.plot(iters_sgd, test_loss_sgd, iters_lbfgs_mb, test_loss_lbfgs_mb,
         iters_lbfgs_fo, test_loss_lbfgs_fo)
plt.xlabel('Iterations')
plt.ylabel('Test Loss')
plt.xlim(iter_lim)
plt.ylim(loss_lim)
plt.title(problem_name)
plt.legend(leg)
if(save):
    plt.savefig(problem_name + '_iters_test_loss_' + label)

# plot iters vs test accuracy
fig8 = plt.figure()
plt.plot(iters_sgd, test_acc_sgd, iters_lbfgs_mb, test_acc_lbfgs_mb,
         iters_lbfgs_fo, test_acc_lbfgs_fo)
plt.xlabel('Iterations')
plt.ylabel('Test Accuracy')
plt.xlim(iter_lim)
plt.ylim(acc_lim)
plt.title(problem_name)
plt.legend(leg)
if(save):
    plt.savefig(problem_name + '_iters_test_acc_' + label)
    
# plot iters and learning rates
fig9 = plt.figure()
plt.plot(iters_sgd, lrs_sgd, iters_lbfgs_mb, lrs_lbfgs_mb, iters_lbfgs_fo, lrs_lbfgs_fo)
plt.xlabel('Iterations')
plt.ylabel('Steplength')
plt.xlim(iter_lim)
plt.title(problem_name)
plt.legend(leg)
if(save):
    plt.savefig(problem_name + '_iters_lrs_' + label)

# plot iters and backtracks
fig10 = plt.figure()
plt.plot(iters_lbfgs_mb, backtracks_lbfgs_mb, iters_lbfgs_fo, backtracks_lbfgs_fo)
plt.xlabel('Iterations')
plt.ylabel('Backtracks')
plt.title(problem_name)
plt.xlim(iter_lim)
plt.legend(leg[1:])
if(save):
    plt.savefig(problem_name + '_iters_backtracks_' + label)
