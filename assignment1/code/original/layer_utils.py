import numpy as np

def affine_forward(x, w):
    

    out = np.reshape(x, (x.shape[0], -1)).dot(w) 
    
    cache = (x, w)

    return out, cache


def affine_backward(dout, cache):

    
    x, w = cache

    N = x.shape[0]

    dx = dout.dot(w.T).reshape(x.shape) # dx = (N, D)

    x_flat = x.reshape(N, -1)
    dw = x_flat.T.dot(dout) # dW = (D, M)

    return dx, dw


def tanh_forward(x):

    out = np.tanh(x)


    cache = x
    return out, cache


def tanh_backward(dout, cache):

    dx, x = None, cache


    dx =  dout * (1 - np.tanh(x) ** 2)

    return dx


def L2_loss(x, y):
    """
    Computes the loss and gradient for L2 loss.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    
    N = x.shape[0]
    C = x.shape[1]
    
    loss = np.sum(np.sum((1/2)*(x - np.eye(C)[y])**2, axis = 1)) / N

    dx =  (x - np.eye(C)[y]) / N

    
    return loss, dx


def affine_tanh_forward(x, w):


    a, fc_cache = affine_forward(x, w)
    
    out, tanh_cache = tanh_forward(a)
    cache = (fc_cache, tanh_cache)
    return out, cache

def affine_tanh_backward(dout, cache):

    fc_cache, tanh_cache = cache
    da = tanh_backward(dout, tanh_cache)

    dx, dw = affine_backward(da, fc_cache)

    return dx, dw

def softmax_loss(x, y):


    N = x.shape[0]

    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    
    loss = -np.log(probs[range(N), y]) 
    loss = np.sum(loss) / N
    
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    
    return loss, dx
