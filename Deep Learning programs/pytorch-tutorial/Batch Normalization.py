"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.4
matplotlib
numpy
"""
import torch
from torch import nn
from torch.nn import init
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np

# torch.manual_seed(1)    # reproducible
# np.random.seed(1)

# Hyper parameters
N_SAMPLES = 2000
BATCH_SIZE = 64
EPOCH = 30
LR = 0.01

# training data
x = np.linspace(-5, 5, N_SAMPLES)[:, np.newaxis]
noise = np.random.normal(0, 1, x.shape)
y = np.square(x) + noise

# test data
test_x = np.linspace(-5, 5, 200)[:, np.newaxis]
noise = np.random.normal(0, 1, test_x.shape)
test_y = np.square(test_x) + noise

train_x, train_y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
test_x = torch.from_numpy(test_x).float()
test_y = torch.from_numpy(test_y).float()

train_dataset = Data.TensorDataset(train_x, train_y)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# show data
plt.scatter(train_x.numpy(), train_y.numpy(), c='#FF9359', s=50, alpha=0.2, label='train')
plt.legend(loc='upper left')



Num_hidden_Neuron = 10
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.hidden1 = nn.Sequential(
            nn.Linear(1, Num_hidden_Neuron),
            nn.BatchNorm1d(Num_hidden_Neuron),
            nn.Tanh()
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(Num_hidden_Neuron, Num_hidden_Neuron),
            nn.BatchNorm1d(Num_hidden_Neuron),
            nn.Tanh()
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(Num_hidden_Neuron, Num_hidden_Neuron),
            nn.BatchNorm1d(Num_hidden_Neuron),
            nn.Tanh()
        )
        self.hidden4 = nn.Sequential(
            nn.Linear(Num_hidden_Neuron, Num_hidden_Neuron),
            nn.BatchNorm1d(Num_hidden_Neuron),
            nn.Tanh()
        )
        self.hidden5 = nn.Sequential(
            nn.Linear(Num_hidden_Neuron, Num_hidden_Neuron),
            nn.BatchNorm1d(Num_hidden_Neuron),
            nn.Tanh()
        )
        self.hidden6 = nn.Sequential(
            nn.Linear(Num_hidden_Neuron, Num_hidden_Neuron),
            nn.BatchNorm1d(Num_hidden_Neuron),
            nn.Tanh()
        )
        self.hidden7 = nn.Sequential(
            nn.Linear(Num_hidden_Neuron, Num_hidden_Neuron),
            nn.BatchNorm1d(Num_hidden_Neuron),
            nn.Tanh()
        )
        self.hidden8 = nn.Sequential(
            nn.Linear(Num_hidden_Neuron, Num_hidden_Neuron),
            nn.BatchNorm1d(Num_hidden_Neuron),
            nn.Tanh()
        )

        # self.hidden1 = nn.Sequential(
        #     nn.Linear(1, Num_hidden_Neuron),
        #     nn.Tanh()
        # )
        # self.hidden2 = nn.Sequential(
        #     nn.Linear(Num_hidden_Neuron, Num_hidden_Neuron),
        #     nn.Tanh()
        # )
        # self.hidden3 = nn.Sequential(
        #     nn.Linear(Num_hidden_Neuron, Num_hidden_Neuron),
        #     nn.Tanh()
        # )
        # self.hidden4 = nn.Sequential(
        #     nn.Linear(Num_hidden_Neuron, Num_hidden_Neuron),
        #     nn.Tanh()
        # )
        # self.hidden5 = nn.Sequential(
        #     nn.Linear(Num_hidden_Neuron, Num_hidden_Neuron),
        #     nn.Tanh()
        # )
        # self.hidden6 = nn.Sequential(
        #     nn.Linear(Num_hidden_Neuron, Num_hidden_Neuron),
        #     nn.Tanh()
        # )
        # self.hidden7 = nn.Sequential(
        #     nn.Linear(Num_hidden_Neuron, Num_hidden_Neuron),
        #     nn.Tanh()
        # )
        # self.hidden8 = nn.Sequential(
        #     nn.Linear(Num_hidden_Neuron, Num_hidden_Neuron),
        #     nn.Tanh()
        # )

        self.predict = nn.Linear(Num_hidden_Neuron, 1)   # output layer

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = self.hidden6(x)
        x = self.hidden7(x)
        x = self.hidden8(x)
        x = self.predict(x)             # linear output
        return x

net = Net()

print(net)    # print net architecture

opt = torch.optim.Adam(net.parameters(), lr=LR)

loss_func = torch.nn.MSELoss()


if __name__ == "__main__":

    # training
    losses = []  # recode loss for two networks

    for epoch in range(EPOCH):
        print('Epoch: ', epoch)

        for step, (b_x, b_y) in enumerate(train_loader):
            pred = net(b_x)
            loss = loss_func(pred, b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()    # it will also learns the parameters in Batch Normalization


        pred = net(test_x)
        # plt.subplot(211)
        plt.plot(test_x.data.numpy(), pred.data.numpy(), c='#74BCFF', lw=4, label='Batch Normalization')
        plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='r', s=50, alpha=0.2, label='train')
        plt.legend(loc='best')
        plt.show()


