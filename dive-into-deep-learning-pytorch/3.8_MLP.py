
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
# torch.manual_seed(1)    # reproducible

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1).cuda()  # x data (tensor), shape=(100, 1)
# y = x.pow(2) + 0.3*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)
y = x.pow(2) + 0.1*torch.randn(x.size()).cuda()              # noisy y data (tensor), shape=(100, 1)


class Net(torch.nn.Module):   # 继承 torch 的 Module 类
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()   # 继承 __init__
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.bn = torch.nn.BatchNorm1d(n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):     # 方法的覆盖(override)，Module 类中的 forward 方法是空的
        x = self.hidden(x)
        x = self.bn(x)
        x = F.relu(x)      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x


net = Net(n_feature=1, n_hidden=100, n_output=1).cuda()     # define the network
print(net)  # net architecture

# optimizer = torch.optim.SGD(net.parameters(), lr=0.2, momentum=0.8)
optimizer = torch.optim.Adam(net.parameters(),lr=0.1)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss


loss_list = []
epoche_list = []
for epoche in range(150):
    prediction = net(x)     # input x and predict based on x

    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients


    # plot and show learning process

    # if epoche % 10 ==0:
    loss_list.append(loss.data.cpu().numpy().tolist())
    epoche_list.append(int(epoche))

    plt.subplot(211)
    plt.plot(x.data.cpu().numpy(), y.data.cpu().numpy(), '.', ms=5)
    plt.plot(x.data.cpu().numpy(), prediction.data.cpu().numpy(), 'r-', lw=1)
    plt.text(0.5, -0.2, 'Loss=%.6f' % loss.data.cpu().numpy(), fontdict={'size': 12, 'color':  'red'})

    plt.subplot(212)
    plt.plot(epoche_list, loss_list, lw=1)
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    # plt.ylim(0, 0.2)
    plt.pause(0.01)

