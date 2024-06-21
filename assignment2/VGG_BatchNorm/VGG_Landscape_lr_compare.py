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
from VGGLL import train_Ls
from models.vgg import VGG_A, VGG_A_Light, VGG_A_Light_BN, VGG_A_Dropout, VGG_A_Dropout_BN
from models.vgg import VGG_A_BatchNorm
from data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
device_id = [0,1,2,3]
num_workers = 4
batch_size = 128

# # add our package dir to path 
# module_path = os.path.dirname(os.getcwd())
# home_path = module_path
# figures_path = os.path.join(home_path, 'reports', 'figures')
# models_path = os.path.join(home_path, 'reports', 'models')

# Make sure you are using the right device.
device_id = device_id
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
# print(device)
# print(torch.cuda.get_device_name(3))


# Initialize your data loader and
# make sure that dataloader works
# as expected by observing one
# sample from it.
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_all_lr(net, lr_list, dir_name = "withoutBN_A", ba = False, num_epoches = 2):
    loss_save_path = 'output/loss/'
    grad_save_path = 'output/grad/'
    model_path = 'output/checkpoint/'
    all_dist = {}
    set_random_seeds(seed_value=2020, device=device)
    criterion = nn.CrossEntropyLoss()
    for lr in lr_list:
        print("running with {}+{}".format(lr, ba))
        model = net()
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        folder_path = str(lr)
        loss_save_path_lr = os.path.join(loss_save_path, dir_name, folder_path)
        grad_save_path_lr = os.path.join(grad_save_path, dir_name, folder_path)
        model_path_lr = os.path.join(model_path, dir_name, folder_path, "model.pth")
        cfg = os.path.join(dir_name, folder_path)
        # # 创建文件夹
        # os.makedirs(loss_save_path_lr, exist_ok=True)
        # os.makedirs(grad_save_path_lr, exist_ok=True)
        # os.makedirs(model_path_lr, exist_ok=True)
        
        max_val_accuracy, max_val_accuracy_epoch, loss = train_Ls(model, 
                                                                        optimizer, criterion, 
                                                                        train_loader, val_loader, cfg = cfg, epochs_n=num_epoches, best_model_path=model_path_lr)
        np.savetxt(os.path.join(loss_save_path_lr, 'loss.txt'), loss, fmt='%s', delimiter=' ')
        # 保存每个 epoch 的梯度
        # for epoch, epoch_grads in enumerate(grads):
        #     flattened_grads = [g.flatten() for g in epoch_grads]  # 将每个梯度展平
        #     np.savetxt(os.path.join(grad_save_path_lr, f'grads_epoch_{epoch + 1}.txt'), flattened_grads, fmt='%s', delimiter=' ')

        # 保存所有梯度
        # all_grads = np.concatenate([np.array(g).flatten() for epoch_grads in grads for g in epoch_grads])
        # np.savetxt(os.path.join(grad_save_path_lr, 'all_grads.txt'), all_grads, fmt='%s', delimiter=' ')
        all_loss = []

        for step_losses in loss:
            if len(step_losses) > 0:
                all_loss += step_losses
        # np.savetxt(os.path.join(loss_save_path_lr, 'all_loss.txt'), all_loss, fmt='%f', delimiter=' ')
        all_dist[lr] = all_loss
        print(f'Max validation accuracy: {max_val_accuracy:.4f} at epoch {max_val_accuracy_epoch + 1}')
    
    max_curve = [max(*values) for values in zip(*[all_dist[i] for i in lr_list])]
    min_curve = [min(*values) for values in zip(*[all_dist[i] for i in lr_list])]
    # 保存 min_curve 和 max_curve
    # np.savetxt(os.path.join(loss_save_path, dir_name, 'min_curve.txt'), min_curve, fmt='%f', delimiter=' ')
    # np.savetxt(os.path.join(loss_save_path, dir_name, 'max_curve.txt'), max_curve, fmt='%f', delimiter=' ')

    return max_curve, min_curve

epoches = 40
max_curve1, min_curve1 = train_all_lr(VGG_A_BatchNorm, [1e-3, 2e-3, 1e-4, 5e-4], dir_name= "withBN_D", ba = True, num_epoches=epoches)
max_curve2, min_curve2 = train_all_lr(VGG_A, [1e-3, 2e-3, 1e-4, 5e-4], dir_name= "withoutBN_D", num_epoches=epoches)

# , 1e-4, 5e-4]
def plot_loss_landscape(min_curve, max_curve, base_path='output/figure',dir_name = "withBN_A"):
    plt.figure(figsize=(10, 6))
    
    # 绘制最小损失曲线和最大损失曲线
    plt.plot(min_curve[10:], label='Min Loss', color='lightpink', alpha=0.3)
    plt.plot(max_curve[10:], label='Max Loss', color='hotpink', alpha=0.3)
    
    # 填充两条曲线之间的区域
    plt.fill_between(range(len(min_curve[10:])), min_curve[10:], max_curve[10:], color='moccasin', alpha=0.5)
    j
    # 添加图例和标签
    plt.legend()
    plt.xlabel('step')
    plt.ylabel('Loss')
    plt.title(f'Loss Landscape ({dir_name})')
    
    # 创建输出路径并保存图形
    plt.savefig(os.path.join(base_path, dir_name, 'loss_landscape_all.png'))
    plt.close()
    
def plot_loss_landscape_compare(min_curve1, max_curve1, min_curve2, max_curve2, base_path='output/figure', dir_name="A"):
    plt.figure(figsize=(10, 6))
    
    # 绘制带BN的最小损失曲线和最大损失曲线
    plt.plot(min_curve1[10:], label='Min Loss with BN', color='lightpink', alpha=0.3)
    plt.plot(max_curve1[10:], label='Max Loss with BN', color='hotpink', alpha=0.3)
    
    # 填充带BN的两条曲线之间的区域
    plt.fill_between(range(len(min_curve1[10:])), min_curve1[10:], max_curve1[10:], color='lightpink', alpha=0.5)
    
    # 绘制不带BN的最小损失曲线和最大损失曲线
    plt.plot(min_curve2[10:], label='Min Loss without BN', color='lightskyblue', alpha=0.3)
    plt.plot(max_curve2[10:], label='Max Loss without BN', color='skyblue', alpha=0.3)
    
    # 填充不带BN的两条曲线之间的区域
    plt.fill_between(range(len(min_curve2[10:])), min_curve2[10:], max_curve2[10:], color='lightskyblue', alpha=0.5)
    
    # 添加图例和标签
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss Landscape with and without BN')
    
    plt.savefig(os.path.join(base_path, dir_name, 'loss_landscape_comparison.png'))
    plt.close()




plot_loss_landscape(min_curve1, max_curve1, base_path='output/figure',dir_name = "withBN_A")
plot_loss_landscape(min_curve2, max_curve2, base_path='output/figure',dir_name = "withoutBN_A")
plot_loss_landscape_compare(min_curve1, max_curve1, min_curve2, max_curve2, base_path='output/figure', dir_name="A")
