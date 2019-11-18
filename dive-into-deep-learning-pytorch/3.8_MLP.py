
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
# torch.manual_seed(1)    # reproducible

x = torch.unsqueeze(torch.linspace(-1, 1, 2000), dim=1).cuda()  # x data (tensor), shape=(100, 1)
# y = x.pow(2) + 0.3*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)
y = x.pow(2) + 0.1*torch.randn(x.size()).cuda()              # noisy y data (tensor), shape=(100, 1)


batch_size = 256

# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(x, y)

# 把 dataset 放入 DataLoader
data_iter = Data.DataLoader(
    dataset=dataset,      # torch TensorDataset format
    batch_size=batch_size,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=0,              # 多线程来读数据, 注意多线程需要在 if __name__ == '__main__': 函数中运行
)
# num_workers=0 表示不用额外的进程来加速读取数据


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
optimizer = torch.optim.Adam(net.parameters(),lr=0.01)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss


loss_list = []
for epoch in range(20):
    for step, (b_x, b_y) in enumerate(data_iter):
        prediction = net(b_x)     # input x and predict based on x

        loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients

        loss_list.append(loss.data.cpu().numpy().tolist())

    prediction = net(x)
    plt.subplot(211)
    plt.plot(x.data.cpu().numpy(), y.data.cpu().numpy(), '.', ms=5)
    plt.plot(x.data.cpu().numpy(), prediction.data.cpu().numpy(), 'r-', lw=1)
    plt.text(0.5, -0.2, 'Loss=%.6f' % loss.data.cpu().numpy(), fontdict={'size': 12, 'color':  'red'})

    plt.subplot(212)
    plt.plot(loss_list, lw=1)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    # plt.ylim(0, 0.2)
    plt.pause(0.01)

