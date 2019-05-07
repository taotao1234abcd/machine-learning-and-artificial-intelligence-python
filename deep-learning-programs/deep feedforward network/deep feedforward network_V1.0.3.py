import torch
import torch.nn as nn
from torch.nn import init
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_validate, KFold
import time

train_mean_error_list = []
test_mean_error_list = []

EPOCH = 400
LR = 0.001
ACTIVATION = torch.sigmoid

out_select = 1

data = pd.read_csv("data/data5000.csv", header=None).values
data_scaler = preprocessing.MinMaxScaler()
data_scaled = data_scaler.fit_transform(data)
data_scaler_y = preprocessing.MinMaxScaler()
data_scaler_y.fit_transform(data[:, out_select:out_select + 1])


class Net(nn.Module):
    def __init__(self, batch_normalization=True):
        super(Net, self).__init__()
        self.do_bn = batch_normalization
        self.fcs = []
        self.bns = []
        self.bn_input = nn.BatchNorm1d(18, momentum=0.5)

        for i in range(N_HIDDEN):
            input_size = 18 if i == 0 else Num_hidden_Neuron
            fc = nn.Linear(input_size, Num_hidden_Neuron)
            setattr(self, 'fc%i' % i, fc)
            self._set_init(fc)
            self.fcs.append(fc)
            if self.do_bn:
                bn = nn.BatchNorm1d(Num_hidden_Neuron, momentum=0.5)
                setattr(self, 'bn%i' % i, bn)
                self.bns.append(bn)

        self.predict = nn.Linear(Num_hidden_Neuron, 1)
        self._set_init(self.predict)

    def _set_init(self, layer):
        init.normal_(layer.weight, mean=0., std=.1)
        init.constant_(layer.bias, 0)

    def forward(self, x):
        pre_activation = [x]
        if self.do_bn:
            x = self.bn_input(x)
        layer_input = [x]
        for i in range(N_HIDDEN):
            x = self.fcs[i](x)
            pre_activation.append(x)
            if self.do_bn:
                x = self.bns[i](x)
            x = ACTIVATION(x)
            layer_input.append(x)
        out = self.predict(x)
        return out


n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True)
kf_times = 0
for index_train, index_test in kf.split(data_scaled):
    kf_times += 1
    data_train_x = data_scaled[index_train, 8:]
    data_train_y = data_scaled[index_train, out_select:out_select + 1]
    data_test_x = data_scaled[index_test, 8:]
    data_test_y = data_scaled[index_test, out_select:out_select + 1]
    if kf_times > 0:
        break

data_train_x = torch.from_numpy(data_train_x).type(torch.FloatTensor).cuda()
data_train_y = torch.from_numpy(data_train_y).type(torch.FloatTensor).cuda()
data_test_x = torch.from_numpy(data_test_x).type(torch.FloatTensor).cuda()
data_test_y = torch.from_numpy(data_test_y).type(torch.FloatTensor).cuda()

LR = 0.001
Num_hidden_Neuron = 500
N_HIDDEN = 6

net = Net().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
loss_func = torch.nn.MSELoss()

time_begin = time.time()
train_loss_list = []
test_loss_list = []
epoch_list = []
for epoch in range(EPOCH):
    if LR > 0.0001:
        LR -= 0.000005
    elif LR > 0.00001:
        LR -= 0.0000005
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    output = net(data_train_x)
    loss = loss_func(output, data_train_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        test_output = net(data_test_x)
        test_loss = loss_func(test_output, data_test_y)
        time_end = time.time()
        print('Epoch:%5d' % (epoch + 1), '| train loss: %.6f' % loss.data.cpu().numpy(),
              '| test loss: %.6f' % test_loss.data.cpu().numpy(),'| time: %.3f' % (time_end - time_begin), 's')
        epoch_list.append(int(epoch + 1))
        train_loss_list.append(loss.data.cpu().numpy().tolist())
        test_loss_list.append(test_loss.data.cpu().numpy().tolist())

        plt.plot(data_test_y.data.cpu().numpy(), label='data test y')
        plt.plot(test_output.data.cpu().numpy(), label='test output')
        plt.legend(loc='best')
        plt.show()


plt.plot(epoch_list, train_loss_list, lw=1, label='Train Loss')
plt.plot(epoch_list, test_loss_list, lw=1, color='red', label='Test Loss')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0, 0.02)
plt.show()


data_train_y_numpy = data_scaler_y.inverse_transform(data_train_y.data.cpu().numpy())
train_output_numpy = data_scaler_y.inverse_transform(output.data.cpu().numpy())
train_mean_error = np.mean(abs((train_output_numpy - data_train_y_numpy)/data_train_y_numpy))
print(' ')
print('平均训练误差:', train_mean_error)

data_test_y_numpy = data_scaler_y.inverse_transform(data_test_y.data.cpu().numpy())
test_output_numpy = data_scaler_y.inverse_transform(test_output.data.cpu().numpy())
test_mean_error = np.mean(abs((test_output_numpy - data_test_y_numpy)/data_test_y_numpy))
print('平均预测误差:', test_mean_error)

