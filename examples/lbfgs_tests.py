"""
L-BFGS Tests

Tests L-BFGS implementation on common unconstrained optimization test problems.

This implementation is CUDA-compatible.

Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
Last edited 8/30/18.

Requirements:
    - NumPy
    - PyTorch

Run Command:
    python full_batch_lbfgs_example.py

"""

import sys
sys.path.append('../functions/')

import torch
import torch.optim
import time

from utils import TestProblem
from LBFGS import LBFGS
import test_problems

#%% Parameters

problem_name = 'Brown'
problem = test_problems.Brown
unique = False
dim = 100

max_iter = 1000
tol = 1e-5
line_search = 'Wolfe'
interpolate = False
max_ls = 100
out = True

#%% Create neural network model
        
model = TestProblem(problem, dim)
if(unique):
    _, x_min, f_min = problem(0, dim, True)
else:
    _, f_min = problem(0, dim, True)

#%% Define optimizer

optimizer = LBFGS(model.parameters(), lr=1, history_size=10, line_search=line_search)

#%% Main training loop

func_evals = 0

t = time.process_time()

optimizer.zero_grad()
obj = model()
obj.backward()
grad = model.grad()
func_evals += 1

# main loop
for n_iter in range(max_iter):
    
    # two-loop recursion to compute search direction
    p = optimizer.two_loop_recursion(-grad)
    
    # define closure for line search
    def closure():              
        optimizer.zero_grad()
        loss_fn = model()
        return loss_fn
    
    # perform line search step
    options = {'closure': closure, 'current_loss': obj, 'eta': 2, 'max_ls': max_ls, 'interpolate': interpolate}
    if(line_search == 'Armijo'):
        obj, lr, backtracks, clos_evals, fail = optimizer.step(p, grad, options)
        
        # compute gradient at new iterate
        obj.backward()
        grad = optimizer._gather_flat_grad()

    elif(line_search == 'Wolfe'):
        obj, grad, lr, backtracks, clos_evals, grad_evals, fail = optimizer.step(p, grad, options)
        
    func_evals += clos_evals
    
    # curvature update
    optimizer.curvature_update(grad)
                
    # print data
    if(out and unique):
        print('Iter:',n_iter+1, 'lr:', lr, 'F - F*:', obj.item() - f_min, 
              '||g||:', torch.norm(grad).item(), '||x - x*||:', torch.norm(model.variables - x_min).item(), 
              'F evals:', func_evals, 'LS Fail:', fail)
    elif(out and not unique):
        print('Iter:',n_iter+1, 'lr:', lr, 'F - F*:', obj.item() - f_min, 
              '||g||:', torch.norm(grad).item(), 'F evals:', func_evals, 'LS Fail:', fail)            
    
    # stopping criterion
    if(torch.norm(grad) < tol):
        break

time = time.process_time() - t    

# print summary
print('============ Summary ============')
print('Problem:', problem_name)
print('Iterations:', n_iter+1)
print('Function Evaluations:', func_evals)
print('Time:', time)
print('F - F*:', obj.item() - f_min)
print('||g||:', torch.norm(grad).item())
if(unique):
    print('||x - x*||:', torch.norm(model.variables - x_min).item())
print('=================================')