
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
from changed.classifiers.fc_net import *
from changed.backprop import Backprop

import time

weight_scale = 1e-1  
learning_rate = 1e-2
solvers = {}
reg = 1e-3
nHidden_layer = [200]

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
    print_every = 500,
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

"""以下是task8的代码"""
# import os
# import pickle
# folder_name = "checkpoint"
# filename = os.path.join("./", folder_name, "/%s.pkl" % ("fine_tuning"))
# dist = {}
# for p, w in best_model.params.items():
#     dist.update({p: w})

# with open(filename, "wb") as f:
#     pickle.dump(dist, f)

# import os
# import pickle
# folder_name = "checkpoint"
# filename = os.path.join("./", folder_name, "/%s.pkl" % ("model1"))
# dist = {}
# for p, w in best_model.params.items():
#     dist.update({p: w})

# file = open(filename, 'wb')
# pickle.dump(dist, file)
# file.close()

    # plt.plot(solver.loss_history)
    # plt.title("Training loss history for nHidden_layer: {}".format(nHidden_layer))
    # plt.xlabel("Iteration")
    # plt.ylabel("Training loss")
    # plt.grid(linestyle='--', linewidth=0.5)
    # plt.savefig("training_loss_history for nHidden_layer.png")

# fig, axes = plt.subplots(3, 1, figsize=(15, 15))

# axes[0].set_title('Training loss')
# axes[0].set_xlabel('Iteration')
# axes[1].set_title('Training accuracy')
# axes[1].set_xlabel('Process')
# axes[2].set_title('Validation accuracy')
# axes[2].set_xlabel('Process')

# for dropout_keep_ratio, solver in solvers.items():
#     axes[0].plot(solver.loss_history, label=f"loss_{dropout_keep_ratio}")
#     axes[1].plot(solver.train_acc_history, label=f"train_acc_{dropout_keep_ratio}")
#     axes[2].plot(solver.val_acc_history, label=f"val_acc_{dropout_keep_ratio}")
    
# for ax in axes:
#     ax.legend(loc="best", ncol=4)
#     ax.grid(linestyle='--', linewidth=0.5)

# plt.show()

# 可视化
# h = y_test_pred.shape
# error_num = []
# for i in range(h[0]):
#     if y_test_pred[i] != ytest2[i]:
#         error_num.append((ytest2[i], y_test_pred[i], i))
        

# import matplotlib.pyplot as plt
# import numpy as np


# fig, axs = plt.subplots((len(error_num) - 1) // 10 + 1, 10, figsize=(20, 8))

# k = 0
# for i in error_num:
#     ax = axs[k // 10, k % 10]
#     ax.imshow(Xtest[i[2]].reshape(16,16), cmap='gray')
#     ax.set_title('true: {}, pred: {}'.format(i[0]+1, i[1]+1))
#     ax.axis('off')
#     k += 1

# plt.savefig('images_error.png')