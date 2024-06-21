
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
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
d = X.shape[1]

# Apply the same transformation to the validation data
Xvalid = (Xvalid - mu) / sigma
Xvalid = np.concatenate((np.ones((Xvalid.shape[0], 1)), Xvalid), axis=1)

# Apply the same transformation to the test data
Xtest = (Xtest - mu) / sigma
Xtest = np.concatenate((np.ones((Xtest.shape[0], 1)), Xtest), axis=1)

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
from original.classifiers.fc_net import *
from original.backprop import Backprop

import time

weight_scale = 1e-1  
learning_rate = 1e-2
solvers = {}
nHidden_layer = [128, 64]
for reg in [1e-3]:
    print('Running with ', reg)
    start_time = time.time()
    model = MLP(
        nHidden_layer,
        input_dim=d,
        reg = reg,
        weight_scale=weight_scale,
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
    solvers["{}".format(reg)] = solver
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

    # plt.plot(solver.loss_history)
    # plt.title("Training loss history for nHidden_layer: {}".format(nHidden_layer))
    # plt.xlabel("Iteration")
    # plt.ylabel("Training loss")
    # plt.grid(linestyle='--', linewidth=0.5)
    # plt.savefig("training_loss_history for nHidden_layer.png")

fig, axes = plt.subplots(3, 1, figsize=(15, 15))

axes[0].set_title('Training loss')
axes[0].set_xlabel('Iteration')
axes[1].set_title('Training accuracy')
axes[1].set_xlabel('Process')
axes[2].set_title('Validation accuracy')
axes[2].set_xlabel('Process')

for reg, solver in solvers.items():
    axes[0].plot(solver.loss_history, label=f"loss_{reg}")
    axes[1].plot(solver.train_acc_history, label=f"train_acc_{reg}")
    axes[2].plot(solver.val_acc_history, label=f"val_acc_{reg}")
    
for ax in axes:
    ax.legend(loc="best", ncol=4)
    ax.grid(linestyle='--', linewidth=0.5)

plt.show()