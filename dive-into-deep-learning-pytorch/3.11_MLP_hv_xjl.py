
import time
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号
import pandas as pd
import torch
from torch import nn
import torch.utils.data as Data
from sklearn import preprocessing
from sklearn.model_selection import KFold

import sys, os
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
# Enable
def enablePrint():
    sys.stdout = sys.__stdout__



USE_FP16 = True  # 模型是否使用半精度浮点数(FP16)

torch.manual_seed(1)

data_x = pd.read_csv("data/hv_xjl/data_x.csv", header=None).values
data_y = pd.read_csv("data/hv_xjl/data_y.csv", header=None).values
data_y = data_y - 1

min_max_scaler = preprocessing.MinMaxScaler()
data_x = min_max_scaler.fit_transform(data_x)

data_x = torch.from_numpy(data_x).type(torch.FloatTensor).cuda()
data_y = torch.from_numpy(data_y).type(torch.int64).cuda()


def train(x_train, y_train, x_valid, y_valid, num_epochs, learning_rate, batch_size):

    num_inputs = 4
    num_outputs = 3

    # 将训练数据的特征和标签组合
    dataset = Data.TensorDataset(x_train, y_train)

    # 把 dataset 放入 DataLoader
    data_iter = Data.DataLoader(
        dataset=dataset,      # torch TensorDataset format
        batch_size=batch_size,      # mini batch size
        shuffle=True,               # 要不要打乱数据 (打乱比较好)
        num_workers=0,              # 多线程来读数据, 注意多线程需要在 if __name__ == '__main__': 函数中运行
    )
    # num_workers=0 表示不用额外的进程来加速读取数据


    net = nn.Sequential(
            nn.Linear(num_inputs, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, num_outputs)
            ).cuda()


    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    blockPrint()
    if USE_FP16 == True:
        from apex import amp
        net, optimizer = amp.initialize(net, optimizer, opt_level="O2")  # 这里是“欧2”，不是“零2”，使用 O1 更稳健
    enablePrint()

    time_begin = time.time()
    train_accuracy_list = []
    test_accuracy_list = []
    step_all_list = []
    step_all = 0
    for epoch in range(num_epochs):
        for step, (b_x, b_y) in enumerate(data_iter):   # gives batch data, normalize x when iterate data_loader
            prediction = net(b_x)
            loss = loss_func(prediction, b_y.squeeze())   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            if USE_FP16 == False:
                loss.backward()  # backpropagation, compute gradients
            else:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            optimizer.step()                # apply gradients

            prediction = net(x_train)
            pred_y = torch.max(prediction, 1)[1].cuda().data
            train_accuracy = torch.sum(pred_y == y_train.squeeze()).type(torch.FloatTensor) / y_train.size(0)
            prediction = net(x_valid)
            pred_y = torch.max(prediction, 1)[1].cuda().data
            accuracy = torch.sum(pred_y == y_valid.squeeze()).type(torch.FloatTensor) / y_valid.size(0)
            time_end = time.time()
            # print('Epoch:%3d' % (epoch + 1), '| step:%5d' % (step + 1), '| train loss: %.4f' % loss.data.cpu().numpy(),
            #       '| train accuracy: %.4f' % train_accuracy, '| test accuracy: %.4f' % accuracy,'| time: %.3f' % (time_end - time_begin), 's')
            step_all_list.append(int(step_all + 1))
            train_accuracy_list.append(train_accuracy)
            test_accuracy_list.append(accuracy)

            step_all += 1

    plt.plot(train_accuracy_list, lw=1)
    plt.plot(test_accuracy_list, lw=1)
    plt.xlabel('Steps')
    plt.legend(['Train Accuracy', 'test Accuracy'])
    plt.show()
    plt.pause(0.01)
    
    return train_accuracy_list, test_accuracy_list



def k_fold(data_x, data_y, num_epochs, learning_rate, batch_size, k=5):

    train_l_sum = 0
    valid_l_sum = 0
    i = 0

    folder = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in folder.split(data_x):
        i += 1
        x_train = data_x[train_index]
        y_train = data_y[train_index]
        x_valid = data_x[test_index]
        y_valid = data_y[test_index]
        train_list, valid_list = train(x_train, y_train, x_valid, y_valid, num_epochs, learning_rate, batch_size)
        train_l_sum += train_list[-1]
        valid_l_sum += valid_list[-1]
        print('fold %d, train accuracy %f, test accuracy %f' % (i, train_list[-1], valid_list[-1]))

    return train_l_sum / k, valid_l_sum / k


num_epochs = 100
lr = 0.01
batch_size = 1024

train_l, valid_l = k_fold(data_x, data_y, num_epochs, lr, batch_size)
print('%d-fold validation: avg train accuracy %f, avg test accuracy %f' % (5, train_l, valid_l))

