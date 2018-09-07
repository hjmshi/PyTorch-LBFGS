"""
Test Problems for Unconstrained Optimization

A set of common unconstrained optimization test problems for testing the L-BFGS
implementation. 

Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
Last edited 9/4/18.

Problems taken from test problems considered by Nash and Nocedal in "A Numerical 
Study of the Limited Memory BFGS Method and the Truncated-Newton Method for Large
Scale Optimization" (1991)

"""

import torch

def ExtendedRosenbrock(x, n, init):
    """
    Extended Rosenbrock function as described in "Testing Unconstrained Optimization
    Software" by More, Garbow, and Hillstrom.
    
    Inputs:
        x (tensor): iterate
        n (int): problem dimension, must be even
        init (bool): initialization flag
            False - returns objective value at iterate f(x)
            True - returns initial iterate x0, minimizer x*, and minimum f*
            
    Outputs:
        f (tensor): objective value
        x0 (tensor): if applicable, initial iterate
        
    """
        
    # check if need initialization and solution
    if(init):
        x0 = torch.ones(n)
        x0[::2] = -1.2
        
        f_min = 0
        x_min = torch.ones(n)
        
        return x0, x_min, f_min

    # check if dimension is even
    if(n % 2 != 0):
        raise ValueError('Problem dimension n must be even!')
    
    # check shape of input
    if(x.shape != torch.Size([n])):
        raise ValueError('Input vector is of wrong size.')
    
    # compute objective function
    f = 0
    for i in range(int(n/2)):
        f += (10*(x[2*(i + 1) - 1] - x[2*(i + 1) - 2]**2))**2
        f += (1 - x[2*(i + 1) - 2])**2
    
    return f

def ExtendedPowell(x, n, init):
    """
    Extended Powell function as described in "Testing Unconstrained Optimization
    Software" by More, Garbow, and Hillstrom.
    
    Inputs:
        x (tensor): iterate
        n (int): problem dimension, must be multiple of 4
        init (bool): initialization flag
            False - returns objective value at iterate f(x)
            True - returns initial iterate x0, minimizer x*, and minimum f*
            
    Outputs:
        f (tensor): objective value
        x0 (tensor): if applicable, initial iterate
        
    """
    
    # check if need initialization and solution
    if(init):
        x0 = torch.ones(n)
        x0[::4] = 3
        x0[1::4] = -1
        x0[2::4] = 0
        
        f_min = 0
        x_min = torch.zeros(n)
        
        return x0, x_min, f_min
        
    # check if dimension is multiple of 4
    if(n % 4 != 0):
        raise ValueError('Problem dimension n must be multiple of 4!')
        
    # check shape of input
    if(x.shape != torch.Size([n])):
        raise ValueError('Input vector is of wrong size.')
        
    # compute objective function
    f = 0
    for i in range(int(n/4)):
        f += (x[4*(i + 1)-4] + 10*x[4*(i+1)-3])**2
        f += 5*(x[4*(i + 1)-2] - x[4*(i + 1)-1])**2
        f += (x[4*(i + 1)-3] - 2*x[4*(i + 1)-2])**4
        f += 10*(x[4*(i + 1)-4] - x[4*(i + 1)-1])**4
        
    return f

def Trigonometric(x, n, init):
    """
    Trigonometric function as described in "Testing Unconstrained Optimization
    Software" by More, Garbow, and Hillstrom.
    
    Inputs:
        x (tensor): iterate
        n (int): problem dimension
        init (bool): initialization flag
            False - returns objective value at iterate f(x)
            True - returns initial iterate x0, objective value f(x0), and minimum f*
            
    Outputs:
        f (tensor): objective value
        x0 (tensor): if applicable, initial iterate
        
    """

    # check if need initialization and solution
    if(init):
        x0 = torch.ones(n)/n
        x = x0
        
        f_min = 0
        
        return x0, f_min
        
    # check shape of input
    if(x.shape != torch.Size([n])):
        raise ValueError('Input vector is of wrong size.')
        
    # compute objective function
    f = 0
    for i in range(n):
        f += (n - torch.sum(torch.cos(x)) + (i + 1)*(1 - torch.cos(x[i])) - torch.sin(x[i]))**2
        
    return f

def VariablyDimensioned(x, n, init):
    """
    Variably dimensioned function as described in "Testing Unconstrained Optimization
    Software" by More, Garbow, and Hillstrom.
    
    Inputs:
        x (tensor): iterate
        n (int): problem dimension
        init (bool): initialization flag
            False - returns objective value at iterate f(x)
            True - returns initial iterate x0, minimizer x*, and minimum f*
            
    Outputs:
        f (tensor): objective value
        x0 (tensor): if applicable, initial iterate
        
    """
    
    # check if need initialization and solution
    if(init):
        x0 = torch.ones(n)
        for j in range(n):
            x0[j] -= (j + 1)/n
        
        f_min = 0
        x_min = torch.ones(n)
        
        return x0, x_min, f_min
                
    # check shape of input
    if(x.shape != torch.Size([n])):
        raise ValueError('Input vector is of wrong size.')
        
    # compute objective function
    f = 0
    f_temp = 0
    for i in range(n):
        f += (x[i] - 1)**2
        f_temp += (i + 1)*(x[i] - 1)
    f += f_temp**2 + f_temp**4
        
    return f

def Brown(x, n, init):
    """
    Brown almost-linear function as described in "Testing Unconstrained Optimization
    Software" by More, Garbow, and Hillstrom.
    
    Inputs:
        x (tensor): iterate
        n (int): problem dimension
        init (bool): initialization flag
            False - returns objective value at iterate f(x)
            True - returns initial iterate x0, minimizer x*, and minimum f*
            
    Outputs:
        f (tensor): objective value
        x0 (tensor): if applicable, initial iterate
        
    """
    
    # check if need initialization and solution
    if(init):
        x0 = 0.5*torch.ones(n)
        f_min = 0
        return x0, f_min
                
    # check shape of input
    if(x.shape != torch.Size([n])):
        raise ValueError('Input vector is of wrong size.')
        
    # compute objective function
    f = 0
    for i in range(n):
        f += (x[i] + torch.sum(x) - (n + 1))**2
    f += (torch.prod(x) - 1)**2
        
    return f

##%% Test Function
#        
#x, f_min = Brown(0, 2, True)
#x.requires_grad_()
#f = Brown(x, 2, False)
#f.backward()
#print(f)
#print(x.grad)
#
#x_min = torch.ones(2, requires_grad=True)
#f = Brown(x_min, 2, False)
#f.backward()
#print(f)
#print(x_min.grad)