import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_saliency_maps(X, y, model):
    model.eval()
    X.requires_grad_()
    X.retain_grad()
    
    score = model(X)
    score = score.gather(1, y.view(-1, 1)).squeeze()
    loss = score.sum()
    loss.backward()
    
    saliency, _ = X.grad.abs().max(1)
    return saliency

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def show_saliency_maps(X, y, model):
    X_tensor = X.to(device)
    y_tensor = y.to(device)

    # 计算 saliency maps
    saliency = compute_saliency_maps(X_tensor, y_tensor, model)
    saliency = saliency.detach().cpu().numpy()

    N = X.shape[0]
    fig, axes = plt.subplots(3, N, figsize=(12, 5))

    for i in range(N):
        img = X[i].detach().numpy().transpose(1, 2, 0)  # 转换为 (H, W, C)
        sal = saliency[i]

        # 标准化 saliency map 以匹配图像的范围
        sal = (sal - sal.min()) / (sal.max() - sal.min())

        # 显示原图
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Label: {classes[y[i]]}")

        axes[1, i].imshow(sal, cmap=plt.cm.hot)
        axes[1, i].axis('off')
        
        # 显示原图和 saliency map 叠加
        axes[2, i].imshow(img)
        axes[2, i].imshow(sal, cmap='hot', alpha=0.5)  # 叠加 saliency map
        axes[2, i].axis('off')

    plt.show()
    
    
def class_visualization_update_step(img, model, target_y, l2_reg, learning_rate):

    score = model(img)
    score = score[:,target_y]
    loss = score - l2_reg * img.norm().sum()
    loss.backward()
    img.data += learning_rate * img.grad / img.grad.norm()
    img.grad.zero_()

def jitter(X, ox, oy):
    """
    Helper function to randomly jitter an image.

    Inputs
    - X: PyTorch Tensor of shape (N, C, H, W)
    - ox, oy: Integers giving number of pixels to jitter along W and H axes

    Returns: A new PyTorch Tensor of shape (N, C, H, W)
    """
    if ox != 0:
        left = X[:, :, :, :-ox]
        right = X[:, :, :, -ox:]
        X = torch.cat([right, left], dim=3)
    if oy != 0:
        top = X[:, :, :-oy]
        bottom = X[:, :, -oy:]
        X = torch.cat([bottom, top], dim=2)
    return X


import random
from scipy.ndimage.filters import gaussian_filter1d
import torchvision.transforms as T

MEAN = np.array([0.5,0.5,0.5], dtype=np.float32)
STD = np.array([0.5,0.5,0.5], dtype=np.float32)


def blur_image(X, sigma=1):
    X_np = X.cpu().clone().numpy()
    X_np = gaussian_filter1d(X_np, sigma, axis=2)
    X_np = gaussian_filter1d(X_np, sigma, axis=3)
    X.copy_(torch.Tensor(X_np).type_as(X))
    return X

def deprocess(img, should_rescale=True):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=(1.0 / STD).tolist()),
        T.Normalize(mean=(-MEAN).tolist(), std=[1, 1, 1]),
        T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x),
        T.ToPILImage(),
    ])
    return transform(img)

def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled

def create_class_visualization(target_y, model, dtype, **kwargs):
    """
    Generate an image to maximize the score of target_y under a pretrained model.

    Inputs:
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - model: A pretrained CNN that will be used to generate the image
    - dtype: Torch datatype to use for computations

    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    - num_iterations: How many iterations to use
    - blur_every: How often to blur the image as an implicit regularizer
    - max_jitter: How much to gjitter the image as an implicit regularizer
    - show_every: How often to show the intermediate result
    """
    model.type(dtype)
    l2_reg = kwargs.pop('l2_reg', 1e-3)
    learning_rate = kwargs.pop('learning_rate', 25)
    num_iterations = kwargs.pop('num_iterations', 100)
    blur_every = kwargs.pop('blur_every', 10)
    max_jitter = kwargs.pop('max_jitter', 16)
    show_every = kwargs.pop('show_every', 25)

    # Randomly initialize the image as a PyTorch Tensor, and make it requires gradient.
    img = torch.randn(1, 3, 224, 224).mul_(1.0).type(dtype).requires_grad_()

    for t in range(num_iterations):
        # Randomly jitter the image a bit; this gives slightly nicer results
        ox, oy = random.randint(0, max_jitter), random.randint(0, max_jitter)
        img.data.copy_(jitter(img.data, ox, oy))
        class_visualization_update_step(img, model, target_y, l2_reg, learning_rate)
        # Undo the random jitter
        img.data.copy_(jitter(img.data, -ox, -oy))

        # As regularizer, clamp and periodically blur the image
        for c in range(3):
            lo = float(-MEAN[c] / STD[c])
            hi = float((1.0 - MEAN[c]) / STD[c])
            img.data[:, c].clamp_(min=lo, max=hi)
        if t % blur_every == 0:
            blur_image(img.data, sigma=0.5)

        # Periodically show the image
        if t == 0 or (t + 1) % show_every == 0 or t == num_iterations - 1:
            plt.imshow(deprocess(img.data.clone().cpu()))
            class_name = classes[target_y]
            plt.title('%s\nIteration %d / %d' % (class_name, t + 1, num_iterations))
            plt.gcf().set_size_inches(4, 4)
            plt.axis('off')
            plt.show()

    return deprocess(img.data.cpu())