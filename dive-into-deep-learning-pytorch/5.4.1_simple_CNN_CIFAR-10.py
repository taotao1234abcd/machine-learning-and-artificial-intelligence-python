
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.utils.data as Data
import torchvision
from sklearn import preprocessing
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']  # ����������ʾ���ı�ǩ
plt.rcParams['axes.unicode_minus']=False  # ����������ʾ����
from PIL import Image

import logging
LOG_FORMAT = "%(asctime)s - %(message)s"
logging.basicConfig(filename=time.strftime("%Y-%m-%d %H%M%S",time.localtime(time.time()))+'.log',
	level=logging.INFO, format=LOG_FORMAT)


def show_images(images, labels):
    _, figs = plt.subplots(6, int(len(images)/6), figsize=(12, 12))
    for i in range(6):
        for j in range(int(len(images)/6)):
            n = int(len(images)/6) * i + j
            f = figs[i, j]
            img = images[n]
            lbl = labels[n]
            img = img.view((3, 32, 32)).numpy().copy()
            img = np.swapaxes(img, 0, 1)
            img = np.swapaxes(img, 1, 2)
            f.imshow(img)
            f.set_title(lbl)
            f.axes.get_xaxis().set_visible(False)
            f.axes.get_yaxis().set_visible(False)
    plt.show()

def get_image(b_x_index, train=False, visual=False):
    # ʹ��PIL��ȡ
    crop = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(32, 32)),
        torchvision.transforms.CenterCrop((31, 31)),
        torchvision.transforms.Resize(size=(32, 32)),
    ])
    n_chose_w = 29 + int(np.random.rand() * 3)
    n_chose_h = 29 + int(np.random.rand() * 3)
    modify = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.RandomCrop((n_chose_w, n_chose_h)),
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(0.1, 0.1, 0.1),
    ])
    loader = torchvision.transforms.ToTensor()
    b_x_index = np.array(b_x_index)
    b_x_index = b_x_index.reshape(-1)
    bx = torch.zeros([len(b_x_index), 3, 32, 32])
    for i in range(len(b_x_index)):
        img_index = b_x_index[i]
        img_pil = Image.open('data/CIFAR-10/train/' + str(img_index) + '.png')         # PIL.Image.Image����
        if train == True:
            img_pil = modify(img_pil)
        else:
            img_pil = crop(img_pil)
        # img = np.array(img_pil)   # (H x W x C), (32 x 32 x 3), [0, 255], RGB
        img = loader(img_pil)   # (C x H x W), (3 x 32 x 32), [0, 1]
        if visual == False:
            img = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        bx[i] = img
    bx = bx.cuda()
    return bx
# b_x_index = np.array([[1], [2], [3]])
# b_x = get_image(b_x_index)

data = pd.read_csv("data/CIFAR-10/trainLabels.csv").values
# data = pd.read_csv("data/CIFAR-10/trainLabels.csv", header=None).values
data_x = data[:, 0:1].copy()
data_y = data[:, 1:2].copy()
del data

data_x = torch.from_numpy(data_x).type(torch.int64)
data_y = torch.from_numpy(data_y).type(torch.int64).cuda()



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(          # input shape (3, 32, 32)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 32, 32)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 16, 16)
        )
        self.conv2 = nn.Sequential(          # input shape (16, 16, 16)
            nn.Conv2d(
                in_channels=16,              # input height
                out_channels=32,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (32, 16, 16)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (32, 8, 8)
        )
        self.out = nn.Linear(32 * 8 * 8, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv to (batch_size, 8 * 8 * 8)
        output = self.out(x)
        return output, x    # return x for visualization


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            x = get_image(x)
            net.eval()
            prediction = net(x)[0]
            net.train()
            pred_y = torch.max(prediction, 1)[1].cuda().data
            acc_sum += torch.sum(pred_y == y.squeeze()).type(torch.FloatTensor)
            n += y.shape[0]
    return acc_sum / n



def train(x_train, y_train, x_valid, y_valid, num_epochs, learning_rate, batch_size):

    # ��ѵ�����ݵ������ͱ�ǩ���
    dataset = Data.TensorDataset(x_train, y_train)
    # �� dataset ���� DataLoader
    data_iter = Data.DataLoader(
        dataset=dataset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  # Ҫ��Ҫ�������� (���ұȽϺ�)
        num_workers=0,  # ���߳���������, ע����߳���Ҫ�� if __name__ == '__main__': ����������
    )
    # num_workers=0 ��ʾ���ö���Ľ��������ٶ�ȡ����


    testset = Data.TensorDataset(x_valid, y_valid)
    test_iter = Data.DataLoader(
        dataset=testset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,  # Ҫ��Ҫ��������
        num_workers=0,  # ���߳���������, ע����߳���Ҫ�� if __name__ == '__main__': ����������
    )


    net = CNN().cuda()
    # print(net)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    time_begin = time.time()
    train_accuracy_list = []
    step_all_list = []
    step_all = 0
    for epoch in range(num_epochs):
        scheduler.step()
        for step, (b_x, b_y) in enumerate(data_iter):  # gives batch data, normalize x when iterate data_loader
            b_x = get_image(b_x, train=True)
            prediction = net(b_x)[0]
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
    pred_labels = net(b_x)[0].argmax(dim=1).cpu().numpy()
    # titles = [str(true) + '\n' + str(pred) for true, pred in zip(true_labels, pred_labels)]
    name = ['�ɻ�', '����', '��', 'è', '¹', '��', '��', '��', '��', '����']
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
lr = 0.002
batch_size = 128

train_l, valid_l = k_fold(data_x, data_y, num_epochs, lr, batch_size)
# print('%d-fold validation: avg train accuracy %f, avg test accuracy %f' % (5, train_l, valid_l))

