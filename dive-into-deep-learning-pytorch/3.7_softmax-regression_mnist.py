
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
import pandas as pd
import torch
from torch import nn
import torch.utils.data as Data

torch.manual_seed(1)


def show_mnist(images, labels):
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()



# train_x = pd.read_csv("data/MNIST/mnist_train_x.csv", header=None).values
# train_y = pd.read_csv("data/MNIST/mnist_train_y.csv", header=None).values
# test_x = pd.read_csv("data/MNIST/mnist_test_x.csv", header=None).values
# test_y = pd.read_csv("data/MNIST/mnist_test_y.csv", header=None).values

train_x = pd.read_csv("data/Fashion-MNIST/fashion-mnist_train_x.csv", header=None).values
train_y = pd.read_csv("data/Fashion-MNIST/fashion-mnist_train_y.csv", header=None).values
test_x = pd.read_csv("data/Fashion-MNIST/fashion-mnist_test_x.csv", header=None).values
test_y = pd.read_csv("data/Fashion-MNIST/fashion-mnist_test_y.csv", header=None).values

train_x = torch.from_numpy(train_x).type(torch.FloatTensor).cuda()
train_x = train_x / 255.0
train_y = torch.from_numpy(train_y).type(torch.int64).cuda()

test_x = torch.from_numpy(test_x).type(torch.FloatTensor).cuda()
test_x = test_x / 255.0
test_y = torch.from_numpy(test_y).type(torch.int64).cuda()


num_inputs = 784
num_outputs = 10


batch_size = 256

# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(train_x, train_y)

# 把 dataset 放入 DataLoader
data_iter = Data.DataLoader(
    dataset=dataset,      # torch TensorDataset format
    batch_size=batch_size,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=0,              # 多线程来读数据, 注意多线程需要在 if __name__ == '__main__': 函数中运行
)
# num_workers=0 表示不用额外的进程来加速读取数据


net = nn.Sequential(
        nn.Linear(num_inputs, num_outputs)

        ).cuda()


loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)


time_begin = time.time()
loss_list = []
accuracy_list = []
step_all_list = []
step_all = 0
for epoch in range(1):
    for step, (b_x, b_y) in enumerate(data_iter):   # gives batch data, normalize x when iterate train_loader
        prediction = net(b_x)
        loss = loss_func(prediction, b_y.squeeze())   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients


        prediction = net(test_x)
        pred_y = torch.max(prediction, 1)[1].cuda().data
        accuracy = torch.sum(pred_y == test_y.squeeze()).type(torch.FloatTensor) / test_y.size(0)
        time_end = time.time()
        print('Epoch:%3d' % (epoch + 1), '| step:%5d' % (step + 1), '| train loss: %.4f' % loss.data.cpu().numpy(),
              '| test accuracy: %.4f' % accuracy,'| time: %.3f' % (time_end - time_begin), 's')
        step_all_list.append(int(step_all + 1))
        loss_list.append(loss.data.cpu().numpy().tolist())
        accuracy_list.append(accuracy)


        plt.subplot(211)
        plt.plot(step_all_list, loss_list, lw=1)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        # plt.ylim(0, 0.5)

        plt.subplot(212)
        plt.plot(step_all_list, accuracy_list, lw=1, color='red')
        plt.xlabel('Steps')
        plt.ylabel('Accuracy')
        # plt.ylim(0.9, 1)
        plt.pause(0.01)

        step_all += 1


true_labels = b_y.cpu().numpy()[:, 0]
pred_labels = net(b_x).argmax(dim=1).cpu().numpy()
# titles = [str(true) + '\n' + str(pred) for true, pred in zip(true_labels, pred_labels)]
name = ['T恤', '裤子', '套头衫', '连衣裙', '大衣', '凉鞋', '衬衫', '运动鞋', '包', '脚踝靴']
titles = [name[int(true)] + '\n' + name[int(pred)] for true, pred in zip(true_labels, pred_labels)]

show_mnist(b_x[0:10].cpu(), titles[0:10])
show_mnist(b_x[10:20].cpu(), titles[10:20])
show_mnist(b_x[20:30].cpu(), titles[20:30])