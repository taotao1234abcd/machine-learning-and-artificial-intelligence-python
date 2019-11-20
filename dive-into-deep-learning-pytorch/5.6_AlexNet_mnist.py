
import time
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号
import pandas as pd
import torch
from torch import nn
import torch.utils.data as Data
import torchvision

torch.manual_seed(1)

import sys, os
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
# Enable
def enablePrint():
    sys.stdout = sys.__stdout__


USE_FP16 = False  # 模型是否使用半精度浮点数(FP16)


def show_mnist(images, labels):
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((224, 224)).numpy())
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


train_x = train_x.reshape(train_x.shape[0], 28, 28)
test_x = test_x.reshape(test_x.shape[0], 28, 28)

train_x = torch.from_numpy(train_x)
train_x = torch.unsqueeze(train_x, dim=1).type(torch.FloatTensor).cuda()
train_x = train_x / 255.0
train_y = torch.from_numpy(train_y).type(torch.int64).cuda()

test_x = torch.from_numpy(test_x)
test_x = torch.unsqueeze(test_x, dim=1).type(torch.FloatTensor).cuda()
test_x = test_x / 255.0
test_y = torch.from_numpy(test_y).type(torch.int64).cuda()


def tensors_resize(data, size):
    crop = torchvision.transforms.Resize(size=size)
    loader = torchvision.transforms.ToTensor()
    unloader = torchvision.transforms.ToPILImage()
    data_get = torch.zeros(data.shape[0], size, size)
    data = data.cpu()
    for i in range(data.shape[0]):
        # print(i)
        temp = unloader(data[i])
        temp = crop(temp)
        temp = loader(temp)
        data_get[i] = temp
    data_get = torch.unsqueeze(data_get, dim=1).type(torch.FloatTensor).cuda()
    return data_get


batch_size = 64

# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(train_x, train_y)

# 把 dataset 放入 DataLoader
data_iter = Data.DataLoader(
    dataset=dataset,            # torch TensorDataset format
    batch_size=batch_size,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=0,              # 多线程来读数据, 注意多线程需要在 if __name__ == '__main__': 函数中运行
)
# num_workers=0 表示不用额外的进程来加速读取数据


testset = Data.TensorDataset(test_x, test_y)
test_iter = Data.DataLoader(
    dataset=testset,            # torch TensorDataset format
    batch_size=batch_size,      # mini batch size
    shuffle=False,              # 要不要打乱数据
    num_workers=0,              # 多线程来读数据, 注意多线程需要在 if __name__ == '__main__': 函数中运行
)


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # kernel_size, stride
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
        self.fc = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        feature = self.conv(x)
        feature = feature.view(x.shape[0], -1)
        output = self.fc(feature)
        return output, feature


net = AlexNet().cuda()


loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)


blockPrint()
if USE_FP16 == True:
    from apex import amp
    net, optimizer = amp.initialize(net, optimizer, opt_level="O1") # 这里是“欧一”，不是“零一”
enablePrint()



def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            x = tensors_resize(x, size=224)
            net.eval()
            prediction = net(x)[0]
            net.train()
            pred_y = torch.max(prediction, 1)[1].cuda().data
            acc_sum += torch.sum(pred_y == y.squeeze()).type(torch.FloatTensor)
            n += y.shape[0]
    return acc_sum / n



time_begin = time.time()
loss_list = []
accuracy_list = []
step_all_list = []
step_all = 0
for epoch in range(5):
    scheduler.step()
    for step, (b_x, b_y) in enumerate(data_iter):   # gives batch data, normalize x when iterate train_loader

        b_x = tensors_resize(b_x, size=224)

        prediction = net(b_x)[0]
        loss = loss_func(prediction, b_y.squeeze())   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step

        if USE_FP16 == False:
            loss.backward()                 # backpropagation, compute gradients
        else:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

        optimizer.step()                # apply gradients

        net.eval()
        prediction = net(b_x)[0]
        net.train()
        pred_y = torch.max(prediction, 1)[1].cuda().data
        accuracy = torch.sum(pred_y == b_y.squeeze()).type(torch.FloatTensor) / b_y.size(0)
        time_end = time.time()
        print('Epoch:%3d' % (epoch + 1), '| step:%5d' % (step + 1), '| train loss: %.4f' % loss.data.cpu().numpy(),
              '| train accuracy: %.4f' % accuracy, '| time: %.3f' % (time_end - time_begin), 's')
        step_all_list.append(int(step_all + 1))
        loss_list.append(loss.data.cpu().numpy().tolist())
        accuracy_list.append(accuracy)

        step_all += 1

    accuracy = evaluate_accuracy(test_iter, net)
    time_end = time.time()
    print('Epoch:%3d' % (epoch + 1), '| step:%5d' % (step + 1), '| train loss: %.4f' % loss.data.cpu().numpy(),
          '| test accuracy: %.4f' % accuracy, '| time: %.3f' % (time_end - time_begin), 's')

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


true_labels = b_y.cpu().numpy()[:, 0]
pred_labels = net(b_x)[0].argmax(dim=1).cpu().numpy()
# titles = [str(true) + '\n' + str(pred) for true, pred in zip(true_labels, pred_labels)]
name = ['T恤', '裤子', '套头衫', '连衣裙', '大衣', '凉鞋', '衬衫', '运动鞋', '包', '脚踝靴']
titles = [name[int(true)] + '\n' + name[int(pred)] for true, pred in zip(true_labels, pred_labels)]

show_mnist(b_x[0:10].cpu(), titles[0:10])
show_mnist(b_x[10:20].cpu(), titles[10:20])
show_mnist(b_x[20:30].cpu(), titles[20:30])


# ======================================================================================================================
# following function (plot_with_labels) is for visualization, can be ignored if not interested
from matplotlib import cm
from sklearn.manifold import TSNE
def plot_with_labels(lowDWeights, labels, labels_text):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s, tx in zip(X, Y, labels, labels_text):
        c = cm.rainbow(int(255 * s / 9))
        plt.text(x, y, tx, backgroundcolor=c, fontsize=6)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize last layer')
    plt.show()
    plt.pause(0.01)
# ======================================================================================================================

x = tensors_resize(test_x[0:500], 224)
net.eval()
prediction, last_layer = net(x)
net.train()

# Visualization of trained flatten layer (T-SNE)
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 500
low_dim_embs = tsne.fit_transform(last_layer.data.cpu().numpy()[:plot_only, :])
labels = test_y.squeeze().cpu().numpy()[:plot_only]
# labels_text = labels
labels_text = [name[int(ind)] for ind in test_y.squeeze().cpu().numpy()[:plot_only]]
plot_with_labels(low_dim_embs, labels, labels_text)

