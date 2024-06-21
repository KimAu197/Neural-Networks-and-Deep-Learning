from builtins import range
from builtins import object
import numpy as np

from ..layer_utils import *


class MLP(object):

    def __init__(self, hidden_dims, input_dim=16*16, num_classes=10, reg=0.0, weight_scale=1e-2, dtype=np.float32, seed=None):
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.reg = reg
        self.params = {}

        layer_dims = np.hstack((input_dim,hidden_dims,num_classes))

        for i in range(self.num_layers):
            W = np.random.normal(0,weight_scale,(layer_dims[i],layer_dims[i+1]))
            self.params['W' + str(i+1)] = W

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):

        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        scores = None

        x = X
        caches = []
        for i in range(self.num_layers -1):
            W = self.params['W'+str(i+1)]

            out,cache = affine_tanh_forward(x,W)
            caches.append(cache)
            x = out
        W = self.params['W'+str(self.num_layers)]
        scores,cache = affine_forward(x,W)
        
        caches.append(cache)

        if mode == "test":
            return scores

        loss, grads = 0.0, {}

        loss, dout = softmax_loss(scores,y)
        for i in range(self.num_layers):
            W = self.params['W' + str(i+1)]
            loss += 0.5 * self.reg * np.sum(W*W)

        dout,dw = affine_backward(dout,caches[self.num_layers - 1])
        # caches[self.num_layers - 1] is the cache of the last layer
        dw += self.reg * self.params['W'+str(self.num_layers)]

        grads['W'+str(self.num_layers)] = dw

        
        for i in range(self.num_layers-2,-1,-1):

            dout, dw = affine_tanh_backward(dout,caches[i])
            dw += self.reg * self.params['W'+str(i+1)]

            grads['W'+str(i+1)] = dw


        return loss, grads
