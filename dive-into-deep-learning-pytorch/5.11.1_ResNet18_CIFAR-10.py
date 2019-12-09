
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
from sklearn import preprocessing
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号
from PIL import Image

import logging
LOG_FORMAT = "%(asctime)s - %(message)s"
logging.basicConfig(filename=time.strftime("%Y-%m-%d %H%M%S",time.localtime(time.time()))+'.log',
	level=logging.INFO, format=LOG_FORMAT)
# logging.info("haha")


def show_images(images, labels):
    _, figs = plt.subplots(6, int(len(images)/6), figsize=(12, 12))
    for i in range(6):
        for j in range(int(len(images)/6)):
            n = int(len(images)/6) * i + j
            f = figs[i, j]
            img = images[n]
            lbl = labels[n]
            img = img.view((3, 224, 224)).numpy().copy()
            img = np.swapaxes(img, 0, 1)
            img = np.swapaxes(img, 1, 2)
            f.imshow(img)
            f.set_title(lbl)
            f.axes.get_xaxis().set_visible(False)
            f.axes.get_yaxis().set_visible(False)
    plt.show()

def get_image(b_x_index, train=False, visual=False):
    # 使用PIL读取
    crop = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.CenterCrop((212, 212)),
        torchvision.transforms.Resize(size=(224, 224)),
    ])
    n_chose_w = 200 + int(np.random.rand() * 24)
    n_chose_h = 200 + int(np.random.rand() * 24)
    modify = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.RandomCrop((n_chose_w, n_chose_h)),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(0.1, 0.1, 0.1),
    ])
    loader = torchvision.transforms.ToTensor()
    b_x_index = np.array(b_x_index)
    b_x_index = b_x_index.reshape(-1)
    bx = torch.zeros([len(b_x_index), 3, 224, 224])
    for i in range(len(b_x_index)):
        img_index = b_x_index[i]
        img_pil = Image.open('data/CIFAR-10/train/' + str(img_index) + '.png')  # PIL.Image.Image对象
        if train == True:
            img_pil = modify(img_pil)
        else:
            img_pil = crop(img_pil)
        # img = np.array(img_pil)   # (H x W x C), (224 x 224 x 3), [0, 255], RGB
        # plt.imshow(img)
        # plt.show()
        img = loader(img_pil)  # (C x H x W), (3 x 224 x 224), [0, 1]
        if visual == False:
            img = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        bx[i] = img
    bx = bx.cuda()
    return bx


data = pd.read_csv("data/CIFAR-10/trainLabels.csv").values
# data = pd.read_csv("data/CIFAR-10/trainLabels.csv", header=None).values
data_x = data[:, 0:1].copy()
data_y = data[:, 1:2].copy()
del data

data_x = torch.from_numpy(data_x).type(torch.int64)
data_y = torch.from_numpy(data_y).type(torch.int64).cuda()


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet18():
    return ResNet(ResidualBlock)


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            x = get_image(x)
            net.eval()
            prediction = net(x)
            net.train()
            pred_y = torch.max(prediction, 1)[1].cuda().data
            acc_sum += torch.sum(pred_y == y.squeeze()).type(torch.FloatTensor)
            n += y.shape[0]
    return acc_sum / n


def train(x_train, y_train, x_valid, y_valid, num_epochs, learning_rate, batch_size):

    # 将训练数据的特征和标签组合
    dataset = Data.TensorDataset(x_train, y_train)
    # 把 dataset 放入 DataLoader
    data_iter = Data.DataLoader(
        dataset=dataset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据, 注意多线程需要在 if __name__ == '__main__': 函数中运行
    )
    # num_workers=0 表示不用额外的进程来加速读取数据


    testset = Data.TensorDataset(x_valid, y_valid)
    test_iter = Data.DataLoader(
        dataset=testset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,  # 要不要打乱数据
        num_workers=0,  # 多线程来读数据, 注意多线程需要在 if __name__ == '__main__': 函数中运行
    )

    net= ResNet18().cuda()

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    time_begin = time.time()
    train_accuracy_list = []
    test_accuracy_list = []
    step_all_list = []
    step_all = 0
    for epoch in range(num_epochs):
        scheduler.step()
        for step, (b_x, b_y) in enumerate(data_iter):  # gives batch data, normalize x when iterate data_loader
            b_x = get_image(b_x, train=True)
            net.eval()
            prediction = net(b_x)
            net.train()
            loss = loss_func(prediction, b_y.squeeze())  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            pred_y = torch.max(prediction, 1)[1].cuda().data
            train_accuracy = torch.sum(pred_y == b_y.squeeze()).type(torch.FloatTensor) / b_y.size(0)
            time_end = time.time()
            # print('Epoch:%3d' % (epoch + 1), '| step:%5d' % (step + 1), '| train loss: %.4f' % loss.data.cpu().numpy(),
            #       '| train accuracy: %.4f' % train_accuracy, '| test accuracy: %.4f' % accuracy,'| time: %.3f' % (time_end - time_begin), 's')
            step_all_list.append(int(step_all + 1))
            train_accuracy_list.append(train_accuracy)

            step_all += 1
        print('Epoch:%3d' % (epoch + 1), '| batch train loss: %.4f' % loss.data.cpu().numpy(),
              '| batch train accuracy: %.4f' % train_accuracy, '| time: %.3f' % (time_end - time_begin), 's')

        if np.mod(epoch, 5) == 4:
            plt.plot(train_accuracy_list, lw=1)
            plt.xlabel('Steps')
            plt.legend(['Batch Train Accuracy'])
            plt.savefig(time.strftime("%Y-%m-%d %H%M%S", time.localtime(time.time())) + '.png', dpi=200)
            plt.show()
            plt.pause(0.01)

            accuracy_train = evaluate_accuracy(data_iter, net)
            accuracy_test = evaluate_accuracy(test_iter, net)

            print('Epoch:%3d, train accuracy: %.4f, test accuracy: %.4f' % (epoch + 1, accuracy_train, accuracy_test))
            logging.info('Epoch:%3d, train accuracy: %.4f, test accuracy: %.4f' % (epoch + 1, accuracy_train, accuracy_test))


    for step, (b_x_i, b_y) in enumerate(test_iter):
        b_x = get_image(b_x_i)
        b_x_v = get_image(b_x_i, visual=True)
        break
    true_labels = b_y.cpu().numpy()[:, 0]
    net.eval()
    prediction = net(b_x)
    net.train()
    pred_labels = torch.max(prediction, 1)[1].cuda().data

    # titles = [str(true) + '\n' + str(pred) for true, pred in zip(true_labels, pred_labels)]
    name = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '蛙', '马', '船', '卡车']
    titles = [name[int(true)] + '\n' + name[int(pred)] for true, pred in zip(true_labels, pred_labels)]

    show_images(b_x_v[0:60].cpu(), titles[0:60])
    show_images(b_x_v[60:120].cpu(), titles[60:120])

    return accuracy_train, accuracy_test


def k_fold(data_x, data_y, num_epochs, learning_rate, batch_size, k=10):
    train_l_sum = 0
    valid_l_sum = 0
    i = 0

    folder = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in folder.split(data_x):
        if i == 0:
            i += 1
        else:
            break
        x_train = data_x[train_index]
        y_train = data_y[train_index]
        x_valid = data_x[test_index]
        y_valid = data_y[test_index]
        accuracy_train, accuracy_test = train(x_train, y_train, x_valid, y_valid, num_epochs, learning_rate, batch_size)
        print('fold %d, train accuracy %f, test accuracy %f' % (i, accuracy_train, accuracy_test))

    return train_l_sum / k, valid_l_sum / k


num_epochs = 30
lr = 0.0012
batch_size = 128

train_l, valid_l = k_fold(data_x, data_y, num_epochs, lr, batch_size)
# print('%d-fold validation: avg train accuracy %f, avg test accuracy %f' % (5, train_l, valid_l))

