import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_validate, KFold
import time


# Hyper Parameters
EPOCH = 50              # train the training data n times
BATCH_SIZE = 500
LR = 0.001              # learning rate

ACTIVATION = nn.Sigmoid


data = pd.read_csv("data/data.csv", header=None).values

data_scaler = preprocessing.MinMaxScaler()
data_scaled = data_scaler.fit_transform(data)

# data_scaler.inverse_transform(data_scaled)


del data
#
# plt.plot(data_scaled)
# plt.show()

kf = KFold(n_splits=10, shuffle=True)

kf_times = 0
for index_train, index_test in kf.split(data_scaled):
    # print("k折划分：%s %s" % (index_train.shape, index_test.shape))
    data_train_x = data_scaled[index_train, 1:]
    data_train_y = data_scaled[index_train, 0:1]
    data_test_x = data_scaled[index_test, 1:]
    data_test_y = data_scaled[index_test, 0:1]
    kf_times += 1
    if kf_times > 0:
        break


data_train_x = torch.from_numpy(data_train_x).type(torch.FloatTensor).cuda()
data_train_y = torch.from_numpy(data_train_y).type(torch.FloatTensor).cuda()
data_test_x = torch.from_numpy(data_test_x).type(torch.FloatTensor).cuda()
data_test_y = torch.from_numpy(data_test_y).type(torch.FloatTensor).cuda()

data_train = Data.TensorDataset(data_train_x, data_train_y)
train_loader = Data.DataLoader(dataset=data_train, batch_size=BATCH_SIZE, shuffle=True)


Num_hidden_Neuron = 1000

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(18, Num_hidden_Neuron),
            nn.BatchNorm1d(Num_hidden_Neuron),
            ACTIVATION()
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(Num_hidden_Neuron, Num_hidden_Neuron),
            nn.BatchNorm1d(Num_hidden_Neuron),
            ACTIVATION()
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(Num_hidden_Neuron, Num_hidden_Neuron),
            nn.BatchNorm1d(Num_hidden_Neuron),
            ACTIVATION()
        )
        self.hidden4 = nn.Sequential(
            nn.Linear(Num_hidden_Neuron, Num_hidden_Neuron),
            nn.BatchNorm1d(Num_hidden_Neuron),
            ACTIVATION()
        )
        self.hidden5 = nn.Sequential(
            nn.Linear(Num_hidden_Neuron, Num_hidden_Neuron),
            nn.BatchNorm1d(Num_hidden_Neuron),
            ACTIVATION()
        )
        self.hidden6 = nn.Sequential(
            nn.Linear(Num_hidden_Neuron, Num_hidden_Neuron),
            nn.BatchNorm1d(Num_hidden_Neuron),
            ACTIVATION()
        )
        self.predict = nn.Linear(Num_hidden_Neuron, 1)   # output layer


    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = self.hidden6(x)
        x = self.predict(x)             # linear output
        return x

net = Net().cuda()
# print(net)

optimizer = torch.optim.Adam(net.parameters(),lr=LR)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss



time_begin = time.time()

# training and testing
train_loss_list = []
test_loss_list = []
step_all_list = []
step_all = 0
for epoch in range(EPOCH):
    if LR > 0.0001:
        LR -= 0.00005
    elif LR > 0.00001:
        LR -= 0.000005
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        output = net(b_x)               # dfn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if (step + 1) % 5 == 0:
            test_output = net(data_test_x)
            test_loss = loss_func(test_output, data_test_y)
            time_end = time.time()
            print('Epoch:%3d' % (epoch + 1), '| step:%5d' % (step_all + 1), '| train loss: %.6f' % loss.data.cpu().numpy(),
                  '| test loss: %.6f' % test_loss.data.cpu().numpy(),'| time: %.3f' % (time_end - time_begin), 's')
            step_all_list.append(int(step_all + 1))
            train_loss_list.append(loss.data.cpu().numpy().tolist())
            test_loss_list.append(test_loss.data.cpu().numpy().tolist())

            # plt.subplot(211)
            # plt.plot(data_test_y.data.cpu().numpy(), label='data test y')
            # plt.plot(test_output.data.cpu().numpy(), label='test output')
            # plt.legend(loc='best')
            #
            # plt.title('Epoch:%3d ' % (epoch + 1)+'| step:%5d ' % (step_all + 1)+
            #           '| test loss: %.6f ' % test_loss.data.cpu().numpy()+ '| time: %.3f' % (time_end - time_begin)+ 's')
            #
            # plt.subplot(212)
            # plt.plot(step_all_list, train_loss_list, lw=1, label='Train Loss')
            # plt.plot(step_all_list, test_loss_list, lw=1, color='red', label='Test Loss')
            # plt.legend(loc='best')
            # plt.xlabel('Steps')
            # plt.ylabel('Loss')
            # plt.ylim(0, 0.02)
            # plt.pause(0.01)

        step_all += 1


plt.plot(data_test_y.data.cpu().numpy(), label='data test y')
plt.plot(test_output.data.cpu().numpy(), label='test output')
plt.legend(loc='best')

plt.title('Epoch:%3d ' % (epoch + 1) + '| step:%5d ' % (step_all + 1) +
          '| test loss: %.6f ' % test_loss.data.cpu().numpy() + '| time: %.3f' % (time_end - time_begin) + 's')
plt.show()

plt.plot(step_all_list, train_loss_list, lw=1, label='Train Loss')
plt.plot(step_all_list, test_loss_list, lw=1, color='red', label='Test Loss')
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim(0, 0.02)
plt.show()
