import sys
import tensorflow as tf
import numpy as np
from six.moves import xrange
 
        
        
# Epsilon Search Parameter
def eps_search(ep, order):
    if(ep == 0.):
        if order == np.inf:
            step_size = 1e-3
            eps = np.arange(1e-3,1e+0,step_size)
        elif order == 2:
            step_size = 1e-2
            eps = np.arange(1e-2,1e+1,step_size)
        elif order == 1:
            step_size = 1e+0
            eps = np.arange(1e+0,1e+3,step_size)
    else:
        eps = [ep]
    return eps
      
        

# Norm Ball Projection
def norm_ball_proj_inner(eta, order, eps):
    if order == np.inf:
        eta = np.clip(eta, -eps, eps)
    elif order in [1, 2]:
        reduc_ind = list(xrange(1, len(eta.shape)))
        if order == 1:
            norm = np.sum(np.abs(eta), axis=tuple(reduc_ind), keepdims=True)
        elif order == 2:
            norm = np.sqrt(np.sum(np.square(eta), axis=tuple(reduc_ind), keepdims=True))
        
        if norm > eps:
            eta = np.multiply(eta, np.divide(eps, norm))
    return eta


# Gradient Normalization            
def grad_normalization(gradients, order):
    if order == np.inf:
        signed_grad = np.sign(gradients)
    elif order in [1, 2]:
        reduc_ind = list(xrange(1, len(gradients.shape)))
        if order == 1:
            norm = np.sum(np.abs(gradients), axis=tuple(reduc_ind), keepdims=True)
        elif order == 2:
            norm = np.sqrt(np.sum(np.square(gradients), axis=tuple(reduc_ind), keepdims=True))    
        signed_grad = gradients / norm
    return signed_grad     


# Get Distortion
def get_dist(a, b, order):    
    if order == np.inf:
        dist = np.max(np.abs(a - b))
    elif order == 1:
        dist = np.sum(np.abs(a - b))
    elif order == 2:
        dist = np.sum((a -b)**2)**.5
    return dist 
      