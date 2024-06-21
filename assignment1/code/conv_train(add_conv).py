
import numpy as np
import scipy.io as sio

# Load data
data = sio.loadmat('./data/digits.mat')
X = data['X']
y = data['y']
Xvalid = data['Xvalid']
yvalid = data['yvalid']
Xtest = data['Xtest']
ytest = data['ytest']

# Get sizes
n, d = X.shape
nLabels = np.max(y)
yExpanded = np.eye(nLabels)[y.flatten() - 1].astype(int)  
t = Xvalid.shape[0]
t2 = Xtest.shape[0]


def standardize_cols(M, mu=None, sigma=None):
    M = M.astype(float)
    if mu is None or sigma is None:
        mu = np.mean(M, axis=0)
        sigma = np.std(M, axis=0)
        sigma[sigma < np.finfo(float).eps] = 1

    S = M - mu
    S = S / sigma.reshape(1, -1)  
    return S, mu, sigma

# Standardize columns and add bias for training data
X, mu, sigma = standardize_cols(X)
d = X.shape[1]

# Apply the same transformation to the validation data
Xvalid = (Xvalid - mu) / sigma

# Apply the same transformation to the test data
Xtest = (Xtest - mu) / sigma

yvalid2 = np.empty((5000,))
for i in range(len(yvalid)):
    yvalid2[i] = yvalid[i][0] - 1


y2 = np.empty((5000,))
for i in range(len(y)):
    y2[i] = y[i][0]-1
    
ytest2 = np.empty((1000,))
for i in range(len(ytest)):
    ytest2[i] = ytest[i][0]-1

y2 = y2.astype(int)
yvalid2 = yvalid2.astype(int)
ytest2 = ytest2.astype(int)

X = X.reshape(n, 1, 16, 16)
Xvalid =  Xvalid.reshape(n, 1, 16, 16)
Xtest = Xtest.reshape(t2, 1, 16, 16)
data = {
  "X_train": X,
  "y_train": y2,
  "X_val": Xvalid,
  "y_val": yvalid2,
}

# Setup cell.
import time
import numpy as np
import matplotlib.pyplot as plt
from conv_main.classifiers.fc_net import *
from conv_main.backprop import Backprop

import time

weight_scale = 1e-1  
learning_rate = 1e-2
solvers = {}
reg = 1e-3
nHidden_layer = [64]

for filter_size in [3]:
    print('Running with ', filter_size)
    start_time = time.time()
    model = MLP_conv(
        nHidden_layer,
        reg = reg,
        dropout_keep_ratio = 1,
        num_filters = 4,
        filter_size = filter_size,
        weight_scale = weight_scale,
        dtype=np.float64
    )
    # import pdb
    # pdb.set_trace()
    solver = Backprop(
        model,
        data,
        update_rule='adam',
        print_every = 100,
        max_iteration = 10000,
        optim_config={"learning_rate": learning_rate},
        lr_decay = 0.95
    )
    solver.train()

    # record best model
    best_val_accuracy = 0
    if solver.best_val_acc > best_val_accuracy:
        best_val_accuracy = solver.best_val_acc
        best_model = model
        


    end_time = time.time()

    y_test_pred = np.argmax(best_model.loss(Xtest), axis=1)
    print('Test set error: ', (y_test_pred != ytest2).mean())

    run_time = end_time - start_time
    print("程序运行时间：", run_time, "秒")
    
import os
import pickle
folder_name = "checkpoint"
filename = os.path.join("./", folder_name, "/%s.pkl" % ("model2"))
dist = {}
for p, w in best_model.params.items():
    dist.update({p: w})

file = open(filename, 'wb')
pickle.dump(dist, file)
file.close()
