import numpy as np
import itertools
def affine_forward(x, w, b):

    out = np.reshape(x, (x.shape[0], -1)).dot(w) + b
    cache = (x, w, b)

    return out, cache


def affine_backward(dout, cache):

    x, w, b = cache
    dx, dw, db = None, None, None

    N = x.shape[0]

    dx = dout.dot(w.T).reshape(x.shape) # dx = (N, D)

    x_flat = x.reshape(N, -1)
    dw = x_flat.T.dot(dout) # dW = (D, M)

    db = np.sum(dout, axis=0) # 1 * dout


    return dx, dw, db


def tanh_forward(x):

    out = np.tanh(x)
    cache = x
    return out, cache


def tanh_backward(dout, cache):

    dx, x = None, cache

    dx =  dout * (1 - np.tanh(x) ** 2)

    return dx


def L2_loss(x, y):
    
    N = x.shape[0]
    C = x.shape[1]
    
    loss = np.sum(np.sum((1/2)*(x - np.eye(C)[y])**2, axis = 1)) / N

    dx =  (x - np.eye(C)[y]) / N

    
    return loss, dx


def affine_tanh_forward(x, w, b):

    a, fc_cache = affine_forward(x, w, b)

    out, tanh_cache = tanh_forward(a)
    cache = (fc_cache, tanh_cache)
    return out, cache

def affine_tanh_backward(dout, cache):

    fc_cache, tanh_cache = cache
    da = tanh_backward(dout, tanh_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    
    return dx, dw, db

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



def dropout_forward(x, dropout_param):

    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])


    if mode == "train":

        mask = (np.random.rand(*x.shape) < p)/p
        out = x * mask

    elif mode == "test":

        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache



def dropout_backward(dout, cache):

    dropout_param, mask = cache
    mode = dropout_param["mode"]

    if mode == "train":

        dx = dout * mask

    elif mode == "test":
        dx = dout
    return dx



def conv_forward(x, w, b, conv_param):
    """
    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the keys:
      - 'stride': The number of pixels between adjacent receptive fields in the

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H - HH) / stride
      W' = 1 + (W - WW) / stride
    - cache: (x, w, b, conv_param)
    """

    stride = conv_param['stride']

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    out = np.zeros((N,F, 1+(H - HH) // stride, 1 + (W -WW) // stride))

    for n in range(N):
        for f in range(F):
            for i in range(0,H - HH + 1,stride):
                for j in range(0, W - WW + 1,stride):
                    out[n,f,i//stride,j//stride]=np.sum(x[n,:,i:i+HH,j:j+WW] * w[f])+b[f]

    cache = (x, w, b, conv_param)
    return out, cache



def conv_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    x,w,b,conv_param = cache
    stride = conv_param['stride']
    
    F,C,HH,WW = w.shape
    N,C,H,W = x.shape

    dw = np.zeros_like(w)
    db = np.sum(dout,axis=(0,2,3))
    dx = np.zeros_like(x)

    for n, f in itertools.product(range(N), range(F)):
        for i, j in itertools.product(range(0,H - HH + 1,stride), range(0,W - WW + 1,stride)):
            dx[n,:,i:i+HH,j:j+WW] += dout[n,f,i//stride,j//stride]*w[f]
            dw[f] += dout[n,f,i//stride,j//stride] * x[n,:,i:i+HH,j:j+WW]

    return dx, dw, db


def conv_tanh_forward(x, w, b, conv_param):

    a, conv_cache = conv_forward(x, w, b, conv_param)
    out, tanh_cache = tanh_forward(a)
    cache = (conv_cache, tanh_cache)
    return out, cache


def conv_tanh_backward(dout, cache):

    conv_cache, tanh_cache = cache
    da = tanh_backward(dout, tanh_cache)
    dx, dw, db = conv_backward(da, conv_cache)
    return dx, dw, db
