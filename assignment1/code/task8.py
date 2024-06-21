import os
import pickle
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

folder_name = "checkpoint"
filename = os.path.join("./", folder_name, "/%s.pkl" % ("fine_tuning"))

with open(filename, "rb") as file:  
    loaded_data = pickle.load(file)

# 得出input
x1 = np.tanh(X.dot(loaded_data["W1"]) + loaded_data["b1"]) # (5000,128)
x2 = np.tanh(x1.dot(loaded_data["W2"]) + loaded_data["b2"]) # (5000,64)

print(x2.shape)
# 使用最小二乘法求解最佳解 W3
y3 = np.eye(10)[y2]


# 使用最小二乘法求解得到W3
W3 = np.linalg.lstsq(x2.T @ x2, x2.T @ y3, rcond=None)[0]

xtest1 = np.tanh(Xtest.dot(loaded_data["W1"]) + loaded_data["b1"])
xtest2 = np.tanh(xtest1.dot(loaded_data["W2"]) + loaded_data["b2"])
xtest_final = xtest2.dot(W3)

y_test_pred = np.argmax(xtest_final, axis=1)
print('Test set error: ', (y_test_pred != ytest2).mean())