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
from VGGLL import train_Gd
from models.vgg import VGG_A, VGG_A_Light, VGG_A_Dropout_BN, VGG_A_Dropout
from models.vgg import VGG_A_BatchNorm, VGG_A_Light_BN
from data.loaders import get_cifar_loader
# ## Constants (parameters) initialization
device_id = [0,1,2,3]
num_workers = 4
batch_size = 128

# add our package dir to path 
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

# Make sure you are using the right device.
device_id = device_id
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(0))


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
        
def train_all_gd(net, lr_list, dir_name = "withoutBN_A", ba = False,num_epoches=2):
    loss_save_path = 'output/loss/'
    grad_save_path = 'output/grad/'
    model_path = 'output/checkpoint/'
    grad_changes_dict = {}
    grads_dict = {}
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
        
        max_val_accuracy, max_val_accuracy_epoch, grads_norm, grads, grad_changes = train_Gd(
            model, optimizer, criterion, train_loader, val_loader, epochs_n=num_epoches,  cfg = cfg, best_model_path=model_path_lr)
        
        grad_changes_dict[lr] = grad_changes
        grads_dict[lr] = grads_norm
        
        
        # 保存每个 epoch 的梯度
        # for epoch, epoch_grads in enumerate(grads):
        #     flattened_grads = [g.flatten() for g in epoch_grads]  # 将每个梯度展平
        #     np.savetxt(os.path.join(grad_save_path_lr, f'grads_epoch_{epoch + 1}.txt'), flattened_grads, fmt='%s', delimiter=' ')
        
        # 保存所有梯度
        all_grads = np.concatenate([np.array(g).flatten() for epoch_grads in grads for g in epoch_grads])
        # np.savetxt(os.path.join(grad_save_path_lr, 'all_grads.txt'), all_grads, fmt='%s', delimiter=' ')
        
    max_curve = [max(*values) for values in zip(*[grads_dict[i] for i in lr_list])]
    min_curve = [min(*values) for values in zip(*[grads_dict[i] for i in lr_list])]
    # 保存 min_curve 和 max_curve
    # np.savetxt(os.path.join(grad_save_path, dir_name, 'min_curve.txt'), min_curve, fmt='%f', delimiter=' ')
    # np.savetxt(os.path.join(grad_save_path, dir_name, 'max_curve.txt'), max_curve, fmt='%f', delimiter=' ')  
    
    return grad_changes_dict, max_curve, min_curve

epoches = 40

# grad_changes_dict1, max_curve1, min_curve1  = train_all_gd(VGG_A_BatchNorm, [1e-3, 2e-3], dir_name= "withBN_A", ba = True,num_epoches=epoches)
# grad_changes_dict2, max_curve2, min_curve2  = train_all_gd(VGG_A, [1e-3, 2e-3], dir_name= "withoutBN_A", num_epoches=epoches)

def plot_grad_changes(grad_changes_dict, base_path='output/figure', dir_name="withoutBN_A"):
    plt.figure(figsize=(10, 6))
    
    colors = ['violet', 'teal', 'springgreen', 'gold']  
    
    for i, (lr, grad_changes) in enumerate(grad_changes_dict.items()):
        plt.plot(grad_changes[10:], label=f'LR={lr} L2 Gradient Changes', color=colors[i % len(colors)], alpha=0.5)  
        
    plt.legend()
    
    # 添加图例和标签
    plt.legend()
    plt.xlabel('step')
    plt.ylabel('L2 Gradient Change')
    plt.title('L2 Gradient Changes for Different Learning Rates')
    
    plt.savefig(os.path.join(base_path, dir_name,  'grad_changes_all.png'))
    plt.close()
    
def plot_Gd_predictness(min_curve, max_curve, base_path='output/figure',dir_name = "withoutBN_A"):
    plt.figure(figsize=(10, 6))
    
    # 绘制最小损失曲线和最大损失曲线
    plt.plot(min_curve[10:], label='Min gd', color='lightskyblue')
    plt.plot(max_curve[10:], label='Max gd', color='salmon')
    
    # 填充两条曲线之间的区域
    plt.fill_between(range(len(min_curve[10:])), min_curve[10:], max_curve[10:], color='lightyellow', alpha=0.5)
    
    # 添加图例和标签
    plt.legend()
    plt.xlabel('step')
    plt.ylabel('Loss')
    plt.title('Gd_predictness')
    
    # 创建输出路径并保存图形
    plt.savefig(os.path.join(base_path, dir_name, 'Gd_predictness.png'))
    plt.close()
    
def plot_gd_compare(min_curve1, max_curve1, min_curve2, max_curve2, base_path='output/figure', dir_name="A"):
    plt.figure(figsize=(10, 6))
    
    # 绘制带BN的最小损失曲线和最大损失曲线
    plt.plot(min_curve1[10:], color='salmon', alpha=0.4)
    plt.plot(max_curve1[10:], label='gd with BN', color='salmon', alpha=0.4)
    
    # 填充带BN的两条曲线之间的区域
    plt.fill_between(range(len(min_curve1[10:])), min_curve1[10:], max_curve1[10:], color='salmon', alpha=0.5)
    
    # 绘制不带BN的最小损失曲线和最大损失曲线
    # plt.plot(min_curve2[1:], label='Min gd without BN', color='lightgreen')
    # plt.plot(max_curve2[1:], label='Max gd without BN', color='lightgreen')
    plt.plot(min_curve2[10:], color='lightgreen', alpha=0.4)
    plt.plot(max_curve2[10:], label='without BN', color='lightgreen', alpha=0.4)
    
    # 填充不带BN的两条曲线之间的区域
    plt.fill_between(range(len(min_curve2[10:])), min_curve2[10:], max_curve2[10:], color='lightgreen', alpha=0.5)
    
    # 添加图例和标签
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel('gd')
    plt.title('gd with and without BN')
    
    plt.savefig(os.path.join(base_path, dir_name, 'gd_pd_comparison.png'))
    plt.close()
    
grad_changes_dict1, max_curve1, min_curve1  = train_all_gd(VGG_A_Dropout_BN, [1e-3, 2e-3, 1e-4, 5e-4], dir_name= "withBN_D", ba = True,num_epoches=epoches)
grad_changes_dict2, max_curve2, min_curve2  = train_all_gd(VGG_A_Dropout, [1e-3, 2e-3, 1e-4, 5e-4], dir_name= "withoutBN_D", num_epoches=epoches)
plot_grad_changes(grad_changes_dict1, base_path='output/figure', dir_name = "withBN_D")
plot_grad_changes(grad_changes_dict2, base_path='output/figure', dir_name = "withoutBN_D")
plot_Gd_predictness(min_curve1, max_curve1, dir_name = "withBN_D")
plot_Gd_predictness(min_curve2, max_curve2, dir_name = "withoutBN_D")
plot_gd_compare(min_curve1, max_curve1, min_curve2, max_curve2, base_path='output/figure', dir_name="D")



