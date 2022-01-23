"""
基线 - 第三步
训练 EfficientNet
"""
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from efficientnet_pytorch import EfficientNet
import os
import time


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=10):
    """Decay learning rate by a f#            model_out_path ="./model/W_epoch_{}.pth".format(epoch)
#            torch.save(model_W, model_out_path) actor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.8 ** (epoch // lr_decay_epoch))
    print('LR is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def train_model(model_ft, criterion, optimizer, lr_scheduler, num_epochs=50):
    train_loss = []
    since = time.time()
    best_model_wts = model_ft.state_dict()
    best_acc = 0.0
    model_ft.train(True)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        optimizer = lr_scheduler(optimizer, epoch)
        running_loss = 0.0
        running_corrects = 0
        count = 0
        for data in train_loader:
            inputs, labels = data
            labels = torch.squeeze(labels.type(torch.LongTensor))
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            outputs = model_ft(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs.data, 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += 1
            if count % 30 == 0 or outputs.size()[0] < BATCH_SIZE:
                print('Epoch:{}: loss:{:.3f}'.format(epoch, loss.item()))
                train_loss.append(loss.item())
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / dset_sizes
        epoch_acc = running_corrects.double() / dset_sizes
        print('Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss, epoch_acc))
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model_ft.state_dict()

    # save best model
    save_dir = 'model'
    os.makedirs(save_dir, exist_ok=True)
    model_ft.load_state_dict(best_model_wts)
    model_out_path = save_dir + "/" + net_name + '.pth'
    torch.save(model_ft, model_out_path)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return train_loss, best_model_wts


# 设置超参数
momentum = 0.9
BATCH_SIZE = 32
class_num = 397
EPOCHS = 500
lr = 0.001
use_gpu = False
net_name = 'efficientnet-b3'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 数据预处理

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
dataset_train = datasets.ImageFolder(r'D:/working/train', transform)
dataset_val = datasets.ImageFolder(r'D:/working/val', transform)
# 对应文件夹的label
print(dataset_train.class_to_idx)
dset_sizes = len(dataset_train)
dset_sizes_val = len(dataset_val)
print("dset_sizes_val Length:", dset_sizes_val)
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True)

model_ft = EfficientNet.from_pretrained('efficientnet-b3')
num_ftrs = model_ft._fc.in_features
model_ft._fc = nn.Linear(num_ftrs, class_num)
criterion = nn.CrossEntropyLoss()
if use_gpu:
    model_ft = model_ft.cuda()
    criterion = criterion.cuda()
optimizer = optim.Adam((model_ft.parameters()), lr=lr)
train_loss, best_model_wts = train_model(model_ft, criterion, optimizer, exp_lr_scheduler, num_epochs=EPOCHS)
