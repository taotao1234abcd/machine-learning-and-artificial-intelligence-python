
"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.4
torchvision
matplotlib
"""

# library
# standard library
import os

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1               # train the training data n times
BATCH_SIZE = 50
LR = 0.001              # learning rate
IMAGE_SIZE = 28

# mnist_train dataset
mnist_train = pd.read_csv("data/mnist_train.csv", header=None).values
train_data_y = mnist_train[:, 0].copy()
train_data_x = mnist_train[:, 1:].copy()
del mnist_train
train_data_x = train_data_x.reshape(train_data_x.shape[0], IMAGE_SIZE, IMAGE_SIZE)

# ==========================================================================================
# plot one example
rand_index = np.random.randint(low=0, high=60000)
plt.imshow(train_data_x[rand_index], cmap='gray')
plt.title('%i' % train_data_y[rand_index])
plt.show()
# ==========================================================================================

train_data_x = train_data_x / 255.0

train_data_y = torch.from_numpy(train_data_y).to(dtype=torch.int64)
train_data_x = torch.from_numpy(train_data_x)
train_data_x = torch.unsqueeze(train_data_x, dim=1).type(torch.FloatTensor)
train_data = Data.TensorDataset(train_data_x, train_data_y)
# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


# mnist_test dataset
mnist_test = pd.read_csv("data/mnist_test.csv", header=None).values
test_data_y = mnist_test[:, 0].copy()
test_data_x = mnist_test[:, 1:].copy()
del mnist_test
test_data_x = test_data_x.reshape(test_data_x.shape[0], IMAGE_SIZE, IMAGE_SIZE)

test_data_x = test_data_x / 255.0

test_data_y = torch.from_numpy(test_data_y).to(dtype=torch.int64)
test_data_x = torch.from_numpy(test_data_x)
test_data_x = torch.unsqueeze(test_data_x, dim=1).type(torch.FloatTensor)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization


cnn = CNN()
print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
                                                        # nn.CrossEntropyLoss() 自带 softmax()函数


# ======================================================================================================================
# following function (plot_with_labels) is for visualization, can be ignored if not interested
from matplotlib import cm
from sklearn.manifold import TSNE
def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)
# ======================================================================================================================

time_begin = time.time()

# training and testing
loss_list = []
accuracy_list = []
step_all_list = []
step_all = 0
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        output = cnn(b_x)[0]               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if (step + 1) % 100 == 0:
            test_output, last_layer = cnn(test_data_x)
            pred_y = torch.max(test_output, 1)[1].data
            accuracy = torch.sum(pred_y == test_data_y).type(torch.FloatTensor) / test_data_y.size(0)
            time_end = time.time()
            print('Epoch:%3d' % (epoch + 1), '| step:%5d' % (step + 1), '| train loss: %.4f' % loss.data.numpy(),
                  '| test accuracy: %.4f' % accuracy,'| time: %.3f' % (time_end - time_begin), 's')
            step_all_list.append(int(step_all + 1))
            loss_list.append(loss.data.numpy().tolist())
            accuracy_list.append(accuracy)

            plt.subplot(211)
            plt.plot(step_all_list, loss_list, lw=1)
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.ylim(0, 0.5)

            plt.subplot(212)
            plt.plot(step_all_list, accuracy_list, lw=1, color='red')
            plt.xlabel('Steps')
            plt.ylabel('Accuracy')
            plt.ylim(0.9, 1)
            plt.pause(0.01)

        step_all += 1


# Visualization of trained flatten layer (T-SNE)
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 500
low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
labels = test_data_y.numpy()[:plot_only]
plot_with_labels(low_dim_embs, labels)


# print predictions from test data
# rand_index = np.random.randint(low=0, high=10000)
# rand_index_range = np.arange(rand_index, (rand_index+35))
# test_output, _ = cnn(test_data_x[rand_index_range])
# pred_y = torch.max(test_output, 1)[1].data.numpy()
# print('index begin:', rand_index)
# print('real number:', test_data_y[rand_index_range].numpy())
# print('prediction: ', pred_y)



