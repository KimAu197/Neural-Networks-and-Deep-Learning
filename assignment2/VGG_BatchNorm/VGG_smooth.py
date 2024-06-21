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
from VGGLL import train_Sm, set_random_seeds
from models.vgg import VGG_A, VGG_A_Light, VGG_A_Dropout_BN, VGG_A_Dropout
from models.vgg import VGG_A_BatchNorm, VGG_A_Light_BN
from data.loaders import get_cifar_loader

train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")


def train_all_sm(net, lr_list, dir_name = "withoutBN_A", ba = False, num_epoches=2):
    loss_save_path = 'output/loss/'
    grad_save_path = 'output/grad/'
    model_path = 'output/checkpoint/'
    max_diff_dict = {}
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
        
        max_diff = train_Sm(model, optimizer, criterion, train_loader, val_loader, 
                              epochs_n=num_epoches,  cfg = cfg, best_model_path=model_path_lr)
        max_diff_dict[lr] = max_diff
        
    smooth_curve = [np.mean(values) for values in zip(*[max_diff_dict[i] for i in lr_list])]
    # val_curve = [np.mean(*values) for values in zip(*[val_acc_dict[i] for i in lr_list])]
    return smooth_curve

epoches = 40

def plot_curve(train_curve1, train_curve2, base_path='output/figure',dir_name = "A"):
    plt.figure(figsize=(10, 6))

    # 绘制最小损失曲线和最大损失曲线
    plt.plot(train_curve1, label='VGG with BN', color='lightcoral', alpha=0.6)
    plt.plot(train_curve2, label='VGG acc without BN', color='lightgreen', alpha=0.4)
    
    # 添加图例和标签
    plt.legend()
    plt.xlabel('step')
    plt.ylabel('β-smoothness')
    plt.title('effective β-smoothness')
    
    # 创建输出路径并保存图形
    plt.savefig(os.path.join(base_path, dir_name, 'effective β-smoothness'))
    plt.close()

smooth_curve1 = train_all_sm(VGG_A_Dropout_BN, 
                                        [1e-3, 2e-3, 1e-4, 5e-4], dir_name= "withBN_D", ba = True, num_epoches=epoches)
smooth_curve2 = train_all_sm(VGG_A_Dropout, 
                                        [1e-3, 2e-3, 1e-4, 5e-4], dir_name= "withoutBN_D", ba = False, num_epoches=epoches)

plot_curve(smooth_curve1, smooth_curve2, base_path="output/figure", dir_name="D")
