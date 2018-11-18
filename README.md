# PyTorch-LBFGS: A PyTorch Implementation of L-BFGS
Authors: Hao-Jun Michael Shi (Northwestern University) and Dheevatsa Mudigere (Facebook)

## What is it?
PyTorch-LBFGS is a modular implementation of L-BFGS, a popular quasi-Newton method, for PyTorch that is compatible with many recent algorithmic advancements
for improving and stabilizing stochastic quasi-Newton methods and addresses many of the deficiencies with the existing
PyTorch L-BFGS implementation. It is designed to provide maximal flexibility to researchers and practitioners in the design and 
implementation of stochastic quasi-Newton methods for training neural networks.

### Main Features
1. Compatible with multi-batch and full-overlap L-BFGS
2. Line searches including (stochastic) Armijo backtracking line search (with or without cubic interpolation) 
and weak Wolfe line search for automatic steplength (or learning rate) selection
3. Powell damping with more sophisticated curvature pair rejection or damping criterion for constructing the quasi-Newton matrix

## Getting Started
To use the L-BFGS optimizer module, simply add `/functions/LBFGS.py` to your current path and use
```
from LBFGS import LBFGS, FullBatchLBFGS
```
to import the L-BFGS or full-batch L-BFGS optimizer, respectively. 

Alternatively, you can add `LBFGS.py` into `torch.optim` on your local PyTorch installation.
To do this, simply add `LBFGS.py` to `/path/to/site-packages/torch/optim`, then modify `/path/to/site-packages/torch/optim/__init__.py`
to include the lines `from LBFGS.py import LBFGS, FullBatchLBFGS` and `del LBFGS, FullBatchLBFGS`. After restarting your Python kernel, 
you will be able to use PyTorch-LBFGS's LBFGS optimizer like any other optimizer in PyTorch.

To see how full-batch, full-overlap, or multi-batch L-BFGS may be easily implemented with a fixed steplength, Armijo backtracking line search, or Wolfe line search, please see the example codes provided in the `/examples/` folder.

## Understanding the Main Features

We give a brief overview of (L-)BFGS and each of the main features of the optimization algorithm. 

### 0. Quasi-Newton Methods
Quasi-Newton methods build an approximation to the Hessian
<a href="https://www.codecogs.com/eqnedit.php?latex=H_k" target="_blank"><img src="https://latex.codecogs.com/svg.latex?H_k" title="H_k" /></a>
to apply a Newton-like algorithm <a href="https://www.codecogs.com/eqnedit.php?latex=x_{k&space;&plus;&space;1}&space;=&space;x_k&space;-&space;\alpha_k&space;H_k&space;\nabla&space;F(x_k)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?x_{k&space;&plus;&space;1}&space;=&space;x_k&space;-&space;\alpha_k&space;H_k&space;\nabla&space;F(x_k)" title="x_{k + 1} = x_k - \alpha_k H_k \nabla F(x_k)" /></a>. 
To do this, it solves for a matrix that satisfies the secant condition
<a href="https://www.codecogs.com/eqnedit.php?latex=H_k&space;(x_k&space;-&space;x_{k&space;-&space;1})&space;=&space;\nabla&space;F(x_k)&space;-&space;\nabla&space;F(x_{k&space;-&space;1})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?H_k&space;(x_k&space;-&space;x_{k&space;-&space;1})&space;=&space;\nabla&space;F(x_k)&space;-&space;\nabla&space;F(x_{k&space;-&space;1})" title="H_k (x_k - x_{k - 1}) = \nabla F(x_k) - \nabla F(x_{k - 1})" /></a>.
**L-BFGS** is one particular optimization algorithm in the family of quasi-Newton methods that approximates the BFGS
algorithm using limited memory. Whereas BFGS requires storing a dense matrix, L-BFGS only requires storing 5-20 vectors to approximate the matrix implicitly and constructs the matrix-vector product on-the-fly via a two-loop recursion. 

In the deterministic or full-batch setting, L-BFGS constructs an approximation to the Hessian by collecting 
**curvature pairs** <a href="https://www.codecogs.com/eqnedit.php?latex=(s_k,&space;y_k)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?(s_k,&space;y_k)" title="(s_k, y_k)" /></a>
defined by differences in consecutive gradients and iterates, i.e.
<a href="https://www.codecogs.com/eqnedit.php?latex=s_k&space;=&space;x_{k&space;&plus;&space;1}&space;-&space;x_k" target="_blank"><img src="https://latex.codecogs.com/svg.latex?s_k&space;=&space;x_{k&space;&plus;&space;1}&space;-&space;x_k" title="s_k = x_{k + 1} - x_k" /></a>
and 
<a href="https://www.codecogs.com/eqnedit.php?latex=y_k&space;=&space;\nabla&space;F(x_{k&space;&plus;&space;1})&space;-&space;\nabla&space;F(x_k)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?y_k&space;=&space;\nabla&space;F(x_{k&space;&plus;&space;1})&space;-&space;\nabla&space;F(x_k)" title="y_k = \nabla F(x_{k + 1}) - \nabla F(x_k)" /></a>
. In our implementation, the curvature pairs are updated after an optimization step is taken (which yields 
<a href="https://www.codecogs.com/eqnedit.php?latex=x_{k&space;&plus;&space;1}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?x_{k&space;&plus;&space;1}" title="x_{k + 1}" /></a>).

Note that other popular optimization methods for deep learning, such as Adam, construct diagonal scalings, whereas L-BFGS constructs a positive definite matrix for scaling the (stochastic) gradient direction.

There are three components to using this algorithm:
1. `two_loop_recursion(vec)`: Applies the L-BFGS two-loop recursion to construct the vector 
<a href="https://www.codecogs.com/eqnedit.php?latex=H_k&space;v" target="_blank"><img src="https://latex.codecogs.com/svg.latex?H_k&space;v" title="H_k v" /></a>.
2. `step(p_k, g_Ok, g_Sk=None, options={})`: Takes a step 
<a href="https://www.codecogs.com/eqnedit.php?latex=x_{k&space;&plus;&space;1}&space;=&space;x_k&space;&plus;&space;\alpha_k&space;p_k" target="_blank"><img src="https://latex.codecogs.com/svg.latex?x_{k&space;&plus;&space;1}&space;=&space;x_k&space;&plus;&space;\alpha_k&space;p_k" title="x_{k + 1} = x_k + \alpha_k p_k" /></a>
and stores <a href="https://www.codecogs.com/eqnedit.php?latex=g_{O_k}&space;=&space;\nabla&space;F_{O_k}(x_k)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?g_{O_k}&space;=&space;\nabla&space;F_{O_k}(x_k)" title="g_{O_k} = \nabla F_{O_k}(x_k)" /></a>
for constructing the next curvature pair. In addition, <a href="https://www.codecogs.com/eqnedit.php?latex=g_{S_k}&space;=&space;\nabla&space;F_{S_k}(x_k)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?g_{S_k}&space;=&space;\nabla&space;F_{S_k}(x_k)" title="g_{S_k} = \nabla F_{S_k}(x_k)" /></a>
is provided to store <a href="https://www.codecogs.com/eqnedit.php?latex=B_k&space;s_k" target="_blank"><img src="https://latex.codecogs.com/svg.latex?B_k&space;s_k" title="B_k s_k" /></a> 
for Powell damping or the curvature pair rejection criterion. (If it is not specified, then <a href="https://www.codecogs.com/eqnedit.php?latex=g_{S_k}&space;=&space;g_{O_k}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?g_{S_k}&space;=&space;g_{O_k}" title="g_{S_k} = g_{O_k}" /></a>.) `options` pass 
necessary parameters or callable functions to the line search.
3. `curvature_update(flat_grad, eps=0.2, damping=True)`: Updates the L-BFGS matrix by computing the curvature pair using `flat_grad`
and the stored <a href="https://www.codecogs.com/eqnedit.php?latex=g_{O_k}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?g_{O_k}" title="g_{O_k}" /></a>
then checks the Powell damping criterion to possibly reject or modify the curvature pair.

If one is interested in using full-batch or deterministic L-BFGS, one can call the `FullBatchLBFGS`optimizer. The `step(options)` function for `FullBatchLBFGS` wraps the two-loop recursion, curvature updating, and step functions from `LBFGS` to simplify the implementation in this case.

Using quasi-Newton methods in the noisy regime requires more work. We will describe below some of the key features of our implementation that will help stabilize L-BFGS when used in conjunction with stochastic gradients. 

### 1. Stable Quasi-Newton Updating
The key to applying quasi-Newton updating in the noisy setting is to require consistency in the gradient difference 
<a href="https://www.codecogs.com/eqnedit.php?latex=y_k" target="_blank"><img src="https://latex.codecogs.com/svg.latex?y_k" title="y_k" /></a> 
in order to prevent differencing noise.

We provide examples of two approaches for doing this:
1. Full-Overlap: This approach simply requires us to evaluate the gradient on a sample twice at both the next and current iterate, 
hence introducing the additional cost of a forward and backward pass over the sample at each iteration, depending on the
line search that is used. In particular, given a sample <a href="https://www.codecogs.com/eqnedit.php?latex=S_k" target="_blank"><img src="https://latex.codecogs.com/svg.latex?S_k" title="S_k" /></a>, we obtain 
<a href="https://www.codecogs.com/eqnedit.php?latex=y_k" target="_blank"><img src="https://latex.codecogs.com/svg.latex?y_k" title="y_k" /></a> 
by computing
<a href="https://www.codecogs.com/eqnedit.php?latex=y_k&space;=&space;\nabla&space;F_{S_k}&space;(x_{k&space;&plus;&space;1})&space;-&space;\nabla&space;F_{S_k}&space;(x_k)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?y_k&space;=&space;\nabla&space;F_{S_k}&space;(x_{k&space;&plus;&space;1})&space;-&space;\nabla&space;F_{S_k}&space;(x_k)" title="y_k = \nabla F_{S_k} (x_{k + 1}) - \nabla F_{S_k} (x_k)" /></a>
.

<p align="center">
<img src="https://github.com/hjmshi/PyTorch-LBFGS/blob/master/figures/full-overlap.png" alt="full-overlap" width="500"/>
</p>

2. Multi-Batch: This approach uses the difference between the gradients over the overlap between two consecutive samples 
<a href="https://www.codecogs.com/eqnedit.php?latex=O_k&space;=&space;S_{k&space;&plus;&space;1}&space;\cap&space;S_k" target="_blank"><img src="https://latex.codecogs.com/svg.latex?O_k&space;=&space;S_{k&space;&plus;&space;1}&space;\cap&space;S_k" title="O_k = S_{k + 1} \cap S_k" /></a>, hence not requiring any additional cost for curvature pair updating, but incurs sampling bias. This approach also suffers
from being generally more tedious to code, although it is more efficient. Note that each sample is the union of the overlap from the previous and current iteration and an additional set of samples, i.e. <a href="https://www.codecogs.com/eqnedit.php?latex=S_k&space;=&space;O_{k&space;-&space;1}&space;\cup&space;N_k&space;\cup&space;O_k" target="_blank"><img src="https://latex.codecogs.com/svg.latex?S_k&space;=&space;O_{k&space;-&space;1}&space;\cup&space;N_k&space;\cup&space;O_k" title="S_k = O_{k - 1} \cup N_k \cup O_k" /></a>. Given two consecutive samples 
<a href="https://www.codecogs.com/eqnedit.php?latex=S_k" target="_blank"><img src="https://latex.codecogs.com/svg.latex?S_k" title="S_k" /></a>
and <a href="https://www.codecogs.com/eqnedit.php?latex=S_{k&space;&plus;&space;1}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?S_{k&space;&plus;&space;1}" title="S_{k + 1}" /></a>, we obtain
<a href="https://www.codecogs.com/eqnedit.php?latex=y_k" target="_blank"><img src="https://latex.codecogs.com/svg.latex?y_k" title="y_k" /></a> 
by computing
<a href="https://www.codecogs.com/eqnedit.php?latex=y_k&space;=&space;\nabla&space;F_{O_k}(x_{k&space;&plus;&space;1})&space;-&space;\nabla&space;F_{O_k}&space;(x_k)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?y_k&space;=&space;\nabla&space;F_{O_k}(x_{k&space;&plus;&space;1})&space;-&space;\nabla&space;F_{O_k}&space;(x_k)" title="y_k = \nabla F_{O_k}(x_{k + 1}) - \nabla F_{O_k} (x_k)" /></a>
. In `multi_batch_lbfgs_example.py`, the variable `g_Ok` denotes 
<a href="https://www.codecogs.com/eqnedit.php?latex=\nabla&space;F_{O_k}&space;(x_k)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nabla&space;F_{O_k}&space;(x_k)" title="\nabla F_{O_k} (x_k)" /></a> 
and the variable `g_Ok_prev` represents
<a href="https://www.codecogs.com/eqnedit.php?latex=\nabla&space;F_{O_k}&space;(x_{k&space;&plus;&space;1})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nabla&space;F_{O_k}&space;(x_{k&space;&plus;&space;1})" title="\nabla F_{O_k} (x_{k + 1})" /></a>
.


<p align="center">
<img src="https://github.com/hjmshi/PyTorch-LBFGS/blob/master/figures/multi-batch.png" alt="multi-batch" width="500"/>
</p>

The code is designed to allow for both of these approaches by delegating control of the samples and the gradients passed to the optimizer to the user. Whereas the existing PyTorch L-BFGS module runs L-BFGS on a fixed sample (possibly full-batch) for a set number of iterations or until convergence, this implementation permits sampling a new mini-batch stochastic gradient at each iteration and is hence amenable with stochastic quasi-Newton methods, and follows the design of other optimizers where one step is equivalent to a single iteration of the algorithm.

### 2. Line Searches
Deterministic quasi-Newton methods, particularly BFGS and L-BFGS, have traditionally been coupled with line searches that 
automatically determine a good steplength (or learning rate) and exploit these well-constructed search directions.
Although these line searches have been crucial to the success of quasi-Newton algorithms in deterministic nonlinear optimization, 
the power of line searches in machine learning have generally been overlooked due to concerns regarding computational cost. 
To overcome these issues, **stochastic** or **probabilistic line searches** have been developed to determine steplengths in
the noisy setting.

We provide four basic (stochastic) line searches that may be used in conjunction with L-BFGS in the step function:
1. (Stochastic) Armijo Backtracking Line Search: Ensures that the Armijo or sufficient decrease condition is satisfied on the function evaluated by the `closure()` function by backtracking from each trial point by multiplying by a constant factor less than 1. 
2. (Stochastic) Armijo Backtracking Line Search with Cubic Interpolation: Similar to the basic backtracking line search
but utilizes a quadratic or cubic interpolation to determine the next trial. This is based on Mark Schmidt's minFunc
MATLAB code.
3. (Stochastic) Weak Wolfe Line Search: Based on Michael Overton's weak Wolfe line search implementation in MATLAB, ensures 
that both the sufficient decrease condition and curvature condition are satisfied on the function evaluated by the `closure()` function by performing a bisection search. 
4. (Stochastic) Weak Wolfe Line Search with Cubic Interpolation: Similar to the weak Wolfe line search but utilizes quadratic interpolation to determine the next trial when backtracking. 

**Note:** For quasi-Newton algorithms, the weak Wolfe line search, although immensely simple, gives similar performance to the strong Wolfe line search, a more complex line search algorithm that utilizes a bracketing and zoom phase, for smooth, nonlinear optimization. In the nonsmooth setting, the weak Wolfe line search (not the strong Wolfe line search) is essential for quasi-Newton algorithms. For these reasons, we only implement a weak Wolfe line search here.

One may also use a constant steplength provided by the user, as in the original PyTorch implementation. See <https://en.wikipedia.org/wiki/Wolfe_conditions> for more detail on the sufficient decrease and curvature conditions.

To use these, when defining the optimizer, the user can specify the line search by setting `line_search` to `Armijo`, `Wolfe`, or `None`.
The user must then define the `options` (typically a closure for reevaluating the model and loss) passed to the step function 
to perform the line search. The `lr` parameter defines the initial steplength in the line search algorithm.

We also provide a `inplace` toggle in the `options` to determine whether or not the variables are updated in-place in the line searches. In-place updating is faster but less numerically accurate than storing the current iterate and reloading after each trial in the
line search.

### 3. Curvature Pair Rejection Criterion and Powell Damping
Another key for L-BFGS is to determine when the history used in constructing the L-BFGS matrix is updated. In particular,
one needs to ensure that the matrix remains positive definite. Existing implementations of L-BFGS have generally checked if
<a href="https://www.codecogs.com/eqnedit.php?latex=y^T&space;s&space;>&space;\epsilon" target="_blank"><img src="https://latex.codecogs.com/svg.latex?y^T&space;s&space;>&space;\epsilon" title="y^T s > \epsilon" /></a>
or <a href="https://www.codecogs.com/eqnedit.php?latex=y^T&space;s&space;>&space;\epsilon&space;\|s\|_2^2" target="_blank"><img src="https://latex.codecogs.com/svg.latex?y^T&space;s&space;>&space;\epsilon&space;\|s\|_2^2" title="y^T s > \epsilon \|s\|_2^2" /></a>,
rejecting the curvature pair if the condition is not satisfied. However, both of these approaches suffer from lack of scale-invariance
of the objective function and reject the curvature pairs when the algorithm is converging close to the solution.

Rather than doing this, we propose using the **Powell damping** condition described in Nocedal and Wright (2006) as the rejection criteria, which ensures
that <a href="https://www.codecogs.com/eqnedit.php?latex=y_k^T&space;s_k&space;>&space;\epsilon&space;s_k^T&space;B_k&space;s_k" target="_blank"><img src="https://latex.codecogs.com/svg.latex?y_k^T&space;s_k&space;>&space;\epsilon&space;s_k^T&space;B_k&space;s_k" title="y_k^T s_k > \epsilon s_k^T B_k s_k" /></a>.
Alternatively, one can modify the definition of <a href="https://www.codecogs.com/eqnedit.php?latex=y_k" target="_blank"><img src="https://latex.codecogs.com/svg.latex?y_k" title="y_k" /></a> 
to ensure that the condition explicitly holds by applying Powell damping to the gradient difference. This has been found
to be useful for the stochastic nonconvex setting.

One can perform curvature pair rejection by setting `damping=False` or apply Powell damping by simply setting `damping=True` 
in the step function. Powell damping is not applied by default.

## Which variant of stochastic L-BFGS should I use?
By default, the algorithm uses a (stochastic) Wolfe line search without Powell damping. 
We recommend implementing this in conjunction with the full-overlap approach with a sufficiently large batch size (say, 2048, 4096, or 8192) as this is easiest to implement and leads to the most stable performance. 
If one uses an Armijo backtracking line search or fixed steplength, we suggest incorporating Powell damping to prevent skipping curvature updates.
Since stochastic quasi-Newton methods are still an active research area, this is by no means the *final* algorithm. We encourage users to try other variants of stochastic L-BFGS to see what works well.

## Recent Changes
We've added the following minor features:
* Full-Batch L-BFGS wrapper.
* Option for in-place updates.
* Quadratic interpolation in Wolfe line search backtracking.

## To Do
In maintaining this module, we are working to add the following features:
* Additional initializations of the L-BFGS matrix aside from the Barzilai-Borwein scaling.
* Wrappers for specific optimizers developed in various papers.
* Using Hessian-vector products for computing curvature pairs.
* More sophisticated stochastic line searches.
* Easy parallelization of L-BFGS methods.

## Acknowledgements
Thanks to Raghu Bollapragada, Jorge Nocedal, and Yuchen Xie for feedback on the details of this implementation, and Kenjy Demeester, Jaroslav Fowkes, and Dominique Orban for help on installing CUTEst and its Python interface for testing the implementation. Many thanks to Jacob Gardner, Geoff Pleiss, and Ben Letham for feedback and help on additional testing on Gaussian processes in GPyTorch.

## References
1. Berahas, Albert S., Jorge Nocedal, and Martin Takác. "A Multi-Batch L-BFGS 
    Method for Machine Learning." Advances in Neural Information Processing 
    Systems. 2016.
2. Bollapragada, Raghu, et al. "A Progressive Batching L-BFGS Method for Machine 
    Learning." International Conference on Machine Learning. 2018.
3. Lewis, Adrian S., and Michael L. Overton. "Nonsmooth Optimization via Quasi-Newton
    Methods." Mathematical Programming 141.1-2 (2013): 135-163.
4. Liu, Dong C., and Jorge Nocedal. "On the Limited Memory BFGS Method for Large Scale Optimization." Mathematical Programming 45.1-3 (1989): 503-528.
5. Nocedal, Jorge. "Updating Quasi-Newton Matrices With Limited Storage." Mathematics of Computation 35.151 (1980): 773-782.
6. Nocedal, Jorge, and Stephen J. Wright. "Numerical Optimization." Springer New York, 2006.
7. Schmidt, Mark. "minFunc: Unconstrained Differentiable Multivariate Optimization 
    in Matlab." Software available at http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html 
    (2005).
8. Schraudolph, Nicol N., Jin Yu, and Simon Günter. "A Stochastic Quasi-Newton 
    Method for Online Convex Optimization." Artificial Intelligence and Statistics. 
    2007.
9. Wang, Xiao, et al. "Stochastic Quasi-Newton Methods for Nonconvex Stochastic 
    Optimization." SIAM Journal on Optimization 27.2 (2017): 927-956.


## Feedback, Questions, or Suggestions?
We are looking for feedback on the code! If you'd like to share your experience with using this module, please let us know here: https://discuss.pytorch.org/t/feedback-on-pytorch-l-bfgs-implementation/26678. 

For more technical issues, questions, or suggestions, please use the Issues tab on the Github repository. Any suggestions on improving the modules are always welcome!
