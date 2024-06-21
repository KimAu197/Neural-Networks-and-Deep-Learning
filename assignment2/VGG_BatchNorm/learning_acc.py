import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display
from VGGLL import train_Ac, set_random_seeds
from models.vgg import VGG_A, VGG_A_Light, VGG_A_Dropout_BN, VGG_A_Dropout
from models.vgg import VGG_A_BatchNorm, VGG_A_Light_BN
from data.loaders import get_cifar_loader

train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

num_epoches = 40
def train_all_ac(net, lr_list, dir_name = "withoutBN_A", ba = False, num_epoches=2):
    loss_save_path = 'output/loss/'
    grad_save_path = 'output/grad/'
    model_path = 'output/checkpoint/'
    train_acc_dict = {}
    val_acc_dict = {}
    set_random_seeds(seed_value=2020, device=device)
    criterion = nn.CrossEntropyLoss()

    for lr in lr_list:
        print("running with {}+{}".format(lr, ba))
        model = net()
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        folder_path = str(lr)
        grad_save_path_lr = os.path.join(grad_save_path, dir_name, folder_path)
        model_path_lr = os.path.join(model_path, dir_name, folder_path, "model.pth")
        cfg = os.path.join(dir_name, folder_path)
        
        train_acc = train_Ac(model, optimizer, criterion, train_loader, val_loader, 
                              epochs_n=num_epoches,  cfg = cfg, best_model_path=model_path_lr)
        train_acc_dict[lr] = train_acc
        
    train_curve = [np.mean(values) for values in zip(*[train_acc_dict[i] for i in lr_list])]
    # val_curve = [np.mean(*values) for values in zip(*[val_acc_dict[i] for i in lr_list])]
    
    return train_curve

epoches = 40

   
def plot_curve(train_curve1, train_curve2, val = False, base_path='output/figure',dir_name = "A"):
    plt.figure(figsize=(10, 6))
    
    if not val:
        # 绘制最小损失曲线和最大损失曲线
        plt.plot(train_curve1, label='train acc with BN', color='cyan')
        plt.plot(train_curve2, label='train acc without BN', color='tomato')
        
        # 填充两条曲线之间的区域
        plt.fill_between(range(len(train_curve1)),train_curve1, train_curve2, color='linen', alpha=0.5)
        
        # 添加图例和标签
        plt.legend()
        plt.xlabel('step')
        plt.ylabel('accuracy')
        plt.title('train learning accuracy')
        
        # 创建输出路径并保存图形
        plt.savefig(os.path.join(base_path, dir_name, 'train learning accuracy'))
        plt.close()
    else:
        # 绘制最小损失曲线和最大损失曲线
        plt.plot(train_curve1, label='val acc with BN', color='cyan')
        plt.plot(train_curve2, label='val acc without BN', color='tomato')
        
        # 填充两条曲线之间的区域
        plt.fill_between(range(len(train_curve1)),train_curve1, train_curve2, color='linen', alpha=0.5)
        
        # 添加图例和标签
        plt.legend()
        plt.xlabel('step')
        plt.ylabel('accuracy')
        plt.title('val learning accuracy')
        
        # 创建输出路径并保存图形
        plt.savefig(os.path.join(base_path, dir_name, 'val learning accuracy'))
        plt.close()

train_curve1 = train_all_ac(VGG_A_Light_BN, 
                                        [1e-3, 2e-3, 1e-4, 5e-4], dir_name= "withBN_L", ba = True,num_epoches=epoches)
train_curve2 = train_all_ac(VGG_A_Light, 
                                        [1e-3, 2e-3, 1e-4, 5e-4], dir_name= "withoutBN_L", ba = False,num_epoches=epoches)

plot_curve(train_curve1, train_curve2, base_path="output/figure", dir_name="L")
# plot_curve(val_curve1, val_curve2, val=True, base_path="output/figure", dir_name="A")


        