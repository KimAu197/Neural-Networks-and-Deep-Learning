from builtins import range
from builtins import object
import numpy as np

from ..layer_utils import *


class MLP(object):

    def __init__(self, hidden_dims, input_dim=16*16, num_classes=10, dropout_keep_ratio=1, reg=0.0, weight_scale=1e-2, dtype=np.float32, seed=None):
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.params = {}

        layer_dims = np.hstack((input_dim,hidden_dims,num_classes))

        for i in range(self.num_layers):
            W = np.random.normal(0,weight_scale,(layer_dims[i],layer_dims[i+1]))
            b = np.zeros(layer_dims[i+1])
            self.params['W' + str(i+1)] = W
            self.params['b' + str(i+1)] = b

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
            
        # dropout
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

    def loss(self, X, y=None):

        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        scores = None

        x = X
        caches = []
        dropout_caches = [] # dropout
        for i in range(self.num_layers -1):
            W = self.params['W'+str(i+1)]
            b = self.params['b'+str(i+1)]
            out,cache = affine_tanh_forward(x, W, b)
            if self.use_dropout:
                out,dropout_cache = dropout_forward(out,self.dropout_param)
                dropout_caches.append(dropout_cache)
            caches.append(cache)
            x = out
        W = self.params['W'+str(self.num_layers)]
        b = self.params['b'+str(self.num_layers)]
        scores,cache = affine_forward(x,W,b)
        caches.append(cache)

        if mode == "test":
            return scores

        loss, grads = 0.0, {}

        # loss, dout = softmax_loss(scores,y)
        loss, dout = L2_loss(scores,y)
        for i in range(self.num_layers):
            W = self.params['W' + str(i+1)]
            loss += 0.5 * self.reg * np.sum(W*W)
            # loss add reg
        dout,dw,db = affine_backward(dout,caches[self.num_layers - 1])
        # caches[self.num_layers - 1] is the cache of the last layer
        dw += self.reg * self.params['W'+str(self.num_layers)]
        # dw add reg
        grads['W'+str(self.num_layers)] = dw
        grads['b'+str(self.num_layers)] = db
        
        for i in range(self.num_layers-2,-1,-1):
            if self.use_dropout:
                dout = dropout_backward(dout,dropout_caches[i])
            dout,dw,db = affine_tanh_backward(dout,caches[i])
            dw += self.reg * self.params['W'+str(i+1)]

            grads['W'+str(i+1)] = dw
            grads['b'+str(i+1)] = db

        return loss, grads
