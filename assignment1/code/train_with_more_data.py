
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
t = Xvalid.shape[0]
t2 = Xtest.shape[0]

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


# 加载保存的数组
more_data1 = np.load('.\data\data_resize.npz')
more_data2 = np.load('.\data\data_noise.npz')

new_data1 = more_data1['new_data1']
new_y1= more_data1['new_y1']
new_data2 = more_data2['new_data2']
new_y2= more_data2['new_y2']

# 垂直合并两个数组
X1 = np.vstack((X, new_data1))
X2 = np.vstack((X, new_data2))
X3 = np.vstack((X2, new_data1))

n1, d = X1.shape
n2, d = X2.shape
n3, d = X3.shape
nLabels = np.max(y)

y21 = np.concatenate((y2, new_y1)).astype(int)
y22 = np.concatenate((y2, new_y2)).astype(int)
y23 = np.concatenate((y22, new_y1)).astype(int)

print("all augmented")

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
X3, mu, sigma = standardize_cols(X3)
d = X3.shape[1]


data = {
  "X_train": X3,
  "y_train": y23,
  "X_val": Xvalid,
  "y_val": yvalid2,
}

# Setup cell.
import time
import numpy as np
import matplotlib.pyplot as plt
from changed.classifiers.fc_net import *
from changed.backprop import Backprop

import time

weight_scale = 1e-1  
learning_rate = 1e-2
solvers = {}
reg = 1e-3
nHidden_layer = [128, 64]

start_time = time.time()
model = MLP(
    nHidden_layer,
    input_dim=d,
    reg = reg,
    dropout_keep_ratio = 1,
    weight_scale = weight_scale,
    dtype=np.float64
)
solver = Backprop(
    model,
    data,
    update_rule='adam',
    print_every = 1500,
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