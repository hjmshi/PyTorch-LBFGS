"""
L-BFGS Tests

Tests L-BFGS implementation on common unconstrained optimization test problems
from the CUTEst test problem set.

Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
Last edited 9/21/18.

Requirements:
    - NumPy
    - PyTorch
    - CUTEst
    - PyCUTEst (https://jfowkes.github.io/pycutest/_build/html/index.html)

Run Command:
    python full_batch_lbfgs_example.py --problemName 'CUTEST_PROBLEM_NAME'

"""

import sys
sys.path.append('../../functions/')

import torch
import torch.optim
import time
import argparse
import pycutest

from utils import CUTEstProblem
from LBFGS import FullBatchLBFGS

#%% Parameters

# parse problem
parser = argparse.ArgumentParser(description='Run L-BFGS on CUTEst problems.')
parser.add_argument('--problemName', type=str, default='L-BFGS',
                    help='CUTEst problem to be solved')
parser.add_argument('--N', type=int, default=0, help='Problem dimension')
args = parser.parse_args()

# define additional parameters
problemName = args.problemName
max_iter = 1000
tol = 1e-3
line_search = 'Wolfe'
interpolate = True
max_ls = 100
history_size = 10
out = True

# find all problems
if problemName == 'ALL':
    problems = pycutest.find_problems(constraints='U', regular=True)
    problems.remove('INDEF')
    Ns = {}
elif problemName == 'L-BFGS':
    problems = ['ARWHEAD', 'BDQRTIC', 'BROYDN7D', 'CRAGGLVY', 'DIXMAANA', 'DIXMAANB',
                'DIXMAANC', 'DIXMAAND', 'DIXMAANE', 'DIXMAANF', 'DIXMAANG', 'DIXMAANH',
                'DIXMAANI', 'DIXMAANK', 'DIXMAANL', 'DQDRTIC', 'DQRTIC', 'DQRTIC',
                'EIGENALS', 'EIGENBLS', 'EIGENCLS', 'ENGVAL1', 'FLETCHBV', 'FREUROTH',
                'GENROSE', 'MOREBV', 'NONDIA', 'NONDQUAR', 'PENALTY1', 'PENALTY3',
                'QUARTC', 'SINQUAD', 'SROSENBR', 'TQUARTIC', 'TRIDIA']
    Ns = {'ARWHEAD': 1000, 'BDQRTIC': 100, 'BROYDN7D': 1000, 'CRAGGLVY': 1000,
          'DIXMAANA': 1500, 'DIXMAANB': 1500, 'DIXMAANC': 1500, 'DIXMAAND': 1500,
          'DIXMAANE': 1500, 'DIXMAANF': 1500, 'DIXMAANG': 1500, 'DIXMAANH': 1500,
          'DIXMAANI': 1500, 'DIXMAANK': 1500, 'DIXMAANL': 1500, 'DQDRTIC': 1000,
          'DQRTIC': 500, 'EIGENALS': 110, 'EIGENBLS': 110, 'EIGENCLS': 462,
          'ENGVAL1': 1000, 'FLETCHBV': 100, 'FREUROTH': 1000, 'GENROSE': 500,
          'MOREBV': 1000, 'NONDIA': 1000, 'NONDQUAR': 100, 'PENALTY1': 1000,
          'PENALTY3': 100, 'QUARTC': 1000, 'SINQUAD': 1000, 'SROSENBR': 1000,
          'TQUARTIC': 1000, 'TRIDIA': 1000}
else:
    problems = [problemName]
    Ns = {problemName: args.N}

#%% Print problem parameters and properties
if problemName != 'ALL' and problemName != 'L-BFGS':
    pycutest.print_available_sif_params(problemName)
    print(pycutest.problem_properties(problemName))
else:
    successes = []
    failures = []
    if(problemName == 'ALL'):
        failures.append('INDEF')

#%% For loop through all problems to solve
for problemName in sorted(problems):

    #%% Create instance of problem
    if problemName in Ns:
        sifParams = {'N': Ns[problemName]}
    else:
        sifParams = {}
    problem = pycutest.import_problem(problemName, sifParams=sifParams)
    model = CUTEstProblem(problem)

    #%% Define optimizer

    optimizer = FullBatchLBFGS(model.parameters(), lr=1, history_size=history_size, line_search=line_search, debug=True)

    #%% Main training loop

    if(out):
        print('===================================================================================')
        print('Solving ' + problemName)
        print('===================================================================================')
        print('    Iter:    |     F       |    ||g||    | |x - y|/|x| |   F Evals   |    alpha    ')
        print('-----------------------------------------------------------------------------------')

    func_evals = 0

    t = time.process_time()

    optimizer.zero_grad()
    obj = model()
    obj.backward()
    grad = model.grad()
    func_evals += 1

    x_old = model.x().clone()
    x_new = x_old.clone()
    f_old = obj

    # main loop
    for n_iter in range(max_iter):

        # define closure for line search
        def closure():
            optimizer.zero_grad()
            loss_fn = model()
            return loss_fn

        # perform line search step
        options = {'closure': closure, 'current_loss': obj, 'eta': 2, 'max_ls': max_ls, 'interpolate': interpolate, 'inplace': False}
        if(line_search == 'Armijo'):
            obj, lr, backtracks, clos_evals, desc_dir, fail = optimizer.step(options=options)

            # compute gradient at new iterate
            obj.backward()
            grad = optimizer._gather_flat_grad()

        elif(line_search == 'Wolfe'):
            obj, grad, lr, backtracks, clos_evals, grad_evals, desc_dir, fail = optimizer.step(options=options)

        x_new.copy_(model.x())

        func_evals += clos_evals

        # compute quantities for checking convergence
        grad_norm = torch.norm(grad)
        x_dist = torch.norm(x_new - x_old)/torch.norm(x_old)
        f_dist = torch.abs(obj - f_old)/torch.max(torch.tensor(1, dtype=torch.float), torch.abs(f_old))

        # print data
        if(out):
            print('  %.3e  |  %.3e  |  %.3e  |  %.3e  |  %.3e  |  %.3e  ' %(n_iter+1, obj.item(), grad_norm.item(), x_dist.item(),  clos_evals, lr))

        # stopping criterion
        if(fail or torch.isnan(obj) or n_iter == max_iter - 1):
            if len(problems) > 1:
                failures.append(problemName)
            break
        elif(torch.norm(grad) < tol or x_dist < 1e-4 or f_dist < 1e-7  or  obj.item() == -float('inf')):
            if len(problems) > 1:
                successes.append(problemName)
            break

        x_old.copy_(x_new)
        f_old.copy_(obj)

    t = time.process_time() - t

    # print summary
    print('==================================== Summary ======================================')
    print('Problem:', problemName)
    print('N:', problem.n)
    print('Iterations:', n_iter+1)
    print('Function Evaluations:', func_evals)
    print('Time:', t)
    print('F:', obj.item())
    print('||g||:', torch.norm(grad).item())
    print('===================================================================================')

# print successful and failed problems
if len(problems) > 1:
    print('Successes:')
    print(successes)
    print('Failures:')
    print(failures)
