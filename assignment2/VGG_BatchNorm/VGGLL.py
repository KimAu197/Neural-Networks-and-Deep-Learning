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
import argparse
from models.vgg import VGG_A, VGG_A_Light, VGG_A_Dropout
from models.vgg import VGG_A_BatchNorm, VGG_A_Light_BN, VGG_A_Dropout_BN
from data.loaders import get_cifar_loader

# parser = argparse.ArgumentParser(description="cfg")

# # 添加参数
# parser.add_argument("--net", default="VGG_A", type=str)
# parser.add_argument("--save_path", default="None", type=str)
# # 解析参数
# args = parser.parse_args()

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
for X, y in train_loader:
    print(X[0])
    label = y[0].item()
    print(y[0])
    print(X[0].shape)
    img = np.transpose(X[0], [1,2,0])
    plt.imshow(img*0.5 + 0.5)
    plt.title(f"Label: {classes[label]}")
    plt.savefig('sample.png')
    plt.savefig('output/figure/sample.png')
    print(X[0].max())
    print(X[0].min())
    break


# This function is used to calculate the accuracy of model classification

def get_accuracy(model, dataloader):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)
    accuracy = total_correct / total_samples
    return accuracy

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

def train_Ls(model, optimizer, criterion, train_loader, val_loader, cfg = None, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    # train_accu_step = []
    # val_accu_step = []
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []
    # grads = []

    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []  # 记录每一步的损失值
        # grad_list = []  # 记录每一步的梯度值
        # learning_curve[epoch] = 0  # 用于绘制训练曲线

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            
            # # 计算训练集准确率
            # train_accuracy = get_accuracy(model, train_loader)
            # train_accu_step.append(train_accuracy)

            # # 计算验证集准确率
            # val_accuracy = get_accuracy(model, val_loader)
            # val_accu_step.append(val_accuracy)
            # 检查损失值
            if torch.isnan(loss).sum() > 0:
                print(f"NaN loss encountered at epoch {epoch + 1}")
                # return learning_curve, train_accuracy_curve, train_accu_step, val_accuracy_curve, val_accu_step, max_val_accuracy, max_val_accuracy_epoch, losses_list, grads
                return max_val_accuracy, max_val_accuracy_epoch, losses_list
            

            # 记录损失值
            loss_list.append(loss.item())
            
            loss.backward()
            
            # 检查梯度值是否爆炸
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         if torch.isnan(param.grad).sum() > 0:
            #             print(f"NaN gradient encountered at {name} at epoch {epoch + 1}")
            #             # return learning_curve, train_accuracy_curve, train_accu_step, val_accuracy_curve, val_accu_step, max_val_accuracy, max_val_accuracy_epoch, losses_list, grads
            #             return  max_val_accuracy, max_val_accuracy_epoch, losses_list, grads
            # grad = model.classifier[4].weight.grad.clone()  # 假设您想记录 classifier[4] 的梯度
            # grad_list.append(grad.cpu().numpy())

            optimizer.step()


        losses_list.append(loss_list)
        # grads.append(grad_list)
        avg_loss = np.mean(loss_list)
        learning_curve[epoch] = avg_loss

        # 计算训练集准确率
        train_accuracy = get_accuracy(model, train_loader)
        train_accuracy_curve[epoch] = train_accuracy

        # 计算验证集准确率
        val_accuracy = get_accuracy(model, val_loader)
        val_accuracy_curve[epoch] = val_accuracy
        
        # 打印当前 epoch 的结果
        print(f"Epoch [{epoch + 1}/{epochs_n}], Loss: {learning_curve[epoch]:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # 如果当前验证准确率是最好的，则保存模型
        if val_accuracy > max_val_accuracy:
            max_val_accuracy = val_accuracy
            max_val_accuracy_epoch = epoch
            if best_model_path:
                torch.save(model.state_dict(), best_model_path)
                
        # display.clear_output(wait=True)
        # f, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # # 绘制学习曲线
        # axes[0].plot(range(1, epoch + 2), learning_curve[:epoch + 1], label='Training Loss')
        # axes[0].set_xlabel('Epoch')
        # axes[0].set_ylabel('Loss')
        # axes[0].set_title('Training Loss Curve')
        # axes[0].legend()

        # # 绘制准确率曲线
        # axes[1].plot(range(1, epoch + 2), train_accuracy_curve[:epoch + 1], label='Training Accuracy')
        # axes[1].plot(range(1, epoch + 2), val_accuracy_curve[:epoch + 1], label='Validation Accuracy')
        # axes[1].set_xlabel('Epoch')
        # axes[1].set_ylabel('Accuracy')
        # axes[1].set_title('Accuracy Curve')
        # axes[1].legend()
        # if cfg:
        #     path = os.path.join('output/figure', cfg, f'training_curve_epoch_{epoch + 1}.png')
        #     plt.savefig(path)
        #     plt.close()
        # else:
        #     plt.savefig(f'output/figure/training_curve_epoch_{epoch + 1}.png')
        #     plt.close()

    return max_val_accuracy, max_val_accuracy_epoch, losses_list

def train_Ac(model, optimizer, criterion, train_loader, val_loader, cfg = None, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    train_accu_step = []
    val_accu_step = []
    max_val_accuracy = 0
    # grads = []

    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            right = (prediction.argmax(1) == y).sum().item()
            accu = right / (y.size(0))

            train_accu_step.append(accu)
            loss.backward()

            optimizer.step()


        # 计算训练集准确率
        train_accuracy = get_accuracy(model, train_loader)
        train_accuracy_curve[epoch] = train_accuracy

        # 计算验证集准确率
        val_accuracy = get_accuracy(model, val_loader)
        val_accuracy_curve[epoch] = val_accuracy
        
        # 打印当前 epoch 的结果
        print(f"Epoch [{epoch + 1}/{epochs_n}], Loss: {learning_curve[epoch]:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # 如果当前验证准确率是最好的，则保存模型
        if val_accuracy > max_val_accuracy:
            max_val_accuracy = val_accuracy
            max_val_accuracy_epoch = epoch
            if best_model_path:
                torch.save(model.state_dict(), best_model_path)

    return train_accu_step

def train_Gd(model, optimizer, criterion, train_loader, val_loader, cfg = None, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    grads = []
    grad_changes = []
    grad_norm = []

    prev_grad = None

    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []  # 记录每一步的损失值
        grad_list = []  # 记录每一步的梯度值
        learning_curve[epoch] = 0  # 用于绘制训练曲线

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            
            loss.backward()
            
            # # 计算当前梯度
            # current_grad = []
            # for p in model.parameters():
            #     if p.grad is not None:
            #         current_grad.append(p.grad.view(-1))
            # current_grad = torch.cat(current_grad)
            
            # if prev_grad is not None:
            #     grad_change = torch.norm(current_grad - prev_grad, p=2).item()
            #     grad_changes.append(grad_change)
            # prev_grad = current_grad.clone()

            grad = model.classifier[4].weight.grad.clone()  # 假设您想记录 classifier[4] 的梯度
            grad_list.append(grad.cpu().numpy())
            grad_norm.append(torch.norm(grad, p=2).item())
            
            current_grad = []
            current_grad.append(grad.view(-1))
            current_grad = torch.cat(current_grad)
            if prev_grad is not None:
                grad_change = torch.norm(current_grad - prev_grad, p=2).item()
                grad_changes.append(grad_change)
            prev_grad = current_grad.clone()
            
            optimizer.step()
        grads.append(grad_list)
        # 计算训练集准确率
        train_accuracy = get_accuracy(model, train_loader)
        train_accuracy_curve[epoch] = train_accuracy

        # 计算验证集准确率
        val_accuracy = get_accuracy(model, val_loader)
        val_accuracy_curve[epoch] = val_accuracy
        
        # 打印当前 epoch 的结果
        print(f"Epoch [{epoch + 1}/{epochs_n}], Loss: {learning_curve[epoch]:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # 如果当前验证准确率是最好的，则保存模型
        if val_accuracy > max_val_accuracy:
            max_val_accuracy = val_accuracy
            max_val_accuracy_epoch = epoch
            if best_model_path:
                torch.save(model.state_dict(), best_model_path)

    return max_val_accuracy, max_val_accuracy_epoch, grad_norm, grads, grad_changes

def train_Sm(model, optimizer, criterion, train_loader, val_loader, cfg = None, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []
    grads = []
    max_grad_diffs = []
    beta_smoothness_list = []

    prev_grads = None

    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []  # 记录每一步的损失值
        grad_list = []  # 记录每一步的梯度值
        learning_curve[epoch] = 0  # 用于绘制训练曲线

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            
            loss.backward()
            
            # 计算当前梯度
            current_grad = []
            current_grad = model.classifier[4].weight.grad.clone()

            # 计算当前梯度与之前所有步的梯度之间的最大差异
            if prev_grads is not None:
                max_diff = torch.norm(current_grad - prev_grads, p=2).item()
                max_grad_diffs.append(max_diff)

                # # 计算 β-smoothness
                # beta_smoothness = max_diff / torch.norm(current_grad, p=2).item()
                # beta_smoothness_list.append(beta_smoothness)
            else:
                max_grad_diffs.append(0)  # 第一次没有前一步梯度
                beta_smoothness_list.append(0)  # 第一次没有前一步梯度

            prev_grads = current_grad.clone()
            optimizer.step()

        # 计算验证集准确率
        val_accuracy = get_accuracy(model, val_loader)
        val_accuracy_curve[epoch] = val_accuracy
        
        # 打印当前 epoch 的结果
        print(f"Epoch [{epoch + 1}/{epochs_n}], Loss: {learning_curve[epoch]:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # 如果当前验证准确率是最好的，则保存模型
        if val_accuracy > max_val_accuracy:
            max_val_accuracy = val_accuracy
            max_val_accuracy_epoch = epoch
            if best_model_path:
                torch.save(model.state_dict(), best_model_path)
    
    return max_grad_diffs
# Train your model
# feel free to modify

# if args.net == "VGG_A":
#     model = VGG_A()
#     if args.save_path:
#         dir = args.save_path
#     else:
#         dir = "withoutBN_A"

# elif args.net == "VGG_A_BN":
#     model = VGG_A_BatchNorm()
#     if args.save_path:
#         dir = args.save_path
#     else:
#         dir = "withBN_A"
        
# elif args.net == "VGG_A_Light":
#     model = VGG_A_Light()
#     if args.save_path:
#         dir = args.save_path
#     else:
#         dir = "withoutBN_L"
    
# elif args.net == "VGG_A_Light_BN":
#     model = VGG_A_Light_BN()
#     if args.save_path:
#         dir = args.save_path
#     else:
#         dir = "withBN_L"

# elif args.net == "VGG_A_Dropout":
#     model = VGG_A_Dropout()
#     if args.save_path:
#         dir = args.save_path
#     else:
#         dir = "withoutBN_D" 

# elif args.net == "VGG_A_Dropout":
#     model = VGG_A_Dropout()
#     if args.save_path:
#         dir = args.save_path
#     else:
#         dir = "withoutBN_D" 
        
# elif args.net == "VGG_A_DBN":
#     model = VGG_A_Dropout_BN()
#     if args.save_path:
#         dir = args.save_path
#     else:
#         dir = "withBN_D" 
    
# print("running with {}".format(args.net))
# dir = "withoutBN_A"
# print("running with {}".format(VGG_A))
# model = VGG_A()
# epo = 3
# set_random_seeds(seed_value=2020, device=device)
# lr = 0.001
# optimizer = torch.optim.Adam(model.parameters(), lr = lr)
# criterion = nn.CrossEntropyLoss()
# loss_save_path = os.path.join('output/loss', dir)
# grad_save_path = os.path.join('output/grad', dir)
# model_path = os.path.join('output/checkpoint', dir, 'model.pth')

# learning_curve, train_accuracy_curve, val_accuracy_curve,max_val_accuracy, max_val_accuracy_epoch, loss, grads = train(model, 
#                                                                                                                       optimizer, criterion, 
#                                                                                                                       train_loader, val_loader, epochs_n=epo, best_model_path=model_path)
# np.savetxt(os.path.join(loss_save_path, 'loss.txt'), loss, fmt='%s', delimiter=' ')
# # 保存每个 epoch 的梯度
# for epoch, epoch_grads in enumerate(grads):
#     flattened_grads = [g.flatten() for g in epoch_grads]  # 将每个梯度展平
#     np.savetxt(os.path.join(grad_save_path, f'grads_epoch_{epoch + 1}.txt'), flattened_grads, fmt='%s', delimiter=' ')

# # 保存所有梯度
# all_grads = np.concatenate([np.array(g).flatten() for epoch_grads in grads for g in epoch_grads])
# np.savetxt(os.path.join(grad_save_path, 'all_grads.txt'), all_grads, fmt='%s', delimiter=' ')

# # Maintain two lists: max_curve and min_curve,
# # select the maximum value of loss in all models
# # on the same step, add it to max_curve, and
# # the minimum value to min_curve
# min_curve = []
# max_curve = []
# ## --------------------
# # Add your code

# for step_losses in loss:
#     if len(step_losses) > 0:
#         min_curve.append(np.min(step_losses))
#         max_curve.append(np.max(step_losses))

# # 保存 min_curve 和 max_curve
# np.savetxt(os.path.join(loss_save_path, 'min_curve.txt'), min_curve, fmt='%f', delimiter=' ')
# np.savetxt(os.path.join(loss_save_path, 'max_curve.txt'), max_curve, fmt='%f', delimiter=' ')

# print(f'Max validation accuracy: {max_val_accuracy:.4f} at epoch {max_val_accuracy_epoch + 1}')
## --------------------

# Use this function to plot the final loss landscape,
# fill the area between the two curves can use plt.fill_between()

def plot_loss_landscape(min_curve, max_curve, cfg = None):

    # min_curve = np.loadtxt(os.path.join(loss_save_path, 'min_curve.txt'))
    # max_curve = np.loadtxt(os.path.join(loss_save_path, 'max_curve.txt'))

    # 创建一个新的图形
    plt.figure(figsize=(10, 6))
    
    # 绘制最小损失曲线和最大损失曲线
    plt.plot(min_curve, label='Min Loss', color='blue')
    plt.plot(max_curve, label='Max Loss', color='red')
    
    # 填充两条曲线之间的区域
    plt.fill_between(range(len(min_curve)), min_curve, max_curve, color='lightgray', alpha=0.5)
    
    # 添加图例和标签
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Landscape')
    if cfg:
        path = os.path.join('output/figure',cfg,'loss_landscape.png')
        plt.savefig(path)
        plt.close
    else:
        plt.savefig('output/figure/loss_landscape.png')
        plt.close()

# # 示例用法
# plot_loss_landscape(min_curve, max_curve, dir)
