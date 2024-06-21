# <center> Project2 </center>

## 1 Train a Network on CIFAR-10 (60%)
CIFAR-10 [4] is a widely used dataset for visual recognition task. The CIFAR-10 dataset (Canadian Institute For
Advanced Research) is a collection of images that are commonly used to train machine learning and computer
vision algorithms. It is one of the most widely used datasets for machine learning research. The CIFAR-10 dataset
contains 60,000 32 × 32 color images in 10 different classes. The 10 different classes represent airplanes, cars, birds,
cats, deer, dogs, frogs, horses, ships, and trucks (as shown in Figure 1). There are 6,000 images of each class. Since
the images in CIFAR-10 are low-resolution (32 × 32), this dataset can allow us to quickly try our models to see
whether it works.
In this project, you will train neural network models on CIFAR-10 to optimize performance. Report the best
test error you are able to achieve on this dataset, and report the structure you constructed to achieve this.

这部分代码在project2.ipynb里

## 2 Batch Normalization (30%)
Batch Normalization (BN) is a widely adopted technique that enables faster and more stable training of deep
neural networks (DNNs). The tendency to improve accuracy and speed up training have established BN as a
favorite technique in deep learning. At a high level, BN is a technique that aims to improve the training of neural
networks by stabilizing the distributions of layer inputs. This is achieved by introducing additional network layers
that control the ﬁrst two moments (mean and variance) of these distributions.
In this project, you will ﬁrst test the eﬀectiveness of BN in the training process, and then explore how does BN
help optimization. The sample codes are provided by Python.

这部分内容在VGG_BatchNorm里