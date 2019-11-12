
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.utils.data as Data
torch.manual_seed(1)

print(torch.__version__)
torch.set_default_tensor_type('torch.FloatTensor')



num_inputs = 1
num_examples = 2000
true_w = 2.5
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1.0, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w * features[:, 0] + true_b
labels += torch.tensor(np.random.normal(0, 1.0, size=labels.size()), dtype=torch.float)


batch_size = 32

# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)

# 把 dataset 放入 DataLoader
data_iter = Data.DataLoader(
    dataset=dataset,      # torch TensorDataset format
    batch_size=batch_size,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=0,              # 多线程来读数据, 注意多线程需要在 if __name__ == '__main__': 函数中运行
)
# num_workers=0 表示不用额外的进程来加速读取数据



# class LinearNet(nn.Module):
#     def __init__(self, n_feature):
#         super(LinearNet, self).__init__()
#         self.linear = nn.Linear(n_feature, 1)
#
#     def forward(self, x):
#         y = self.linear(x)
#         return y

# net = LinearNet(num_inputs)


net = nn.Sequential(
    nn.Linear(num_inputs, 1)

    )


loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)


loss_list = []
epoch_list = []
num_epochs = 30
for epoch in range(1, num_epochs + 1):
    for x, y in data_iter:
        prediction = net(x)
        loss = loss_func(prediction, y.view(-1, 1))
        optimizer.zero_grad()  # 梯度清零
        loss.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, loss))

    loss_list.append(loss.data.numpy().tolist())
    epoch_list.append(int(epoch))

    xx = torch.unsqueeze(torch.linspace(-4, 4, 1000), dim=1)
    prediction = net(xx)
    plt.subplot(211)
    plt.plot(features.data.numpy(), labels.data.numpy(), '.', ms=3)
    plt.plot(xx.data.numpy(), prediction.data.numpy(), 'r-', lw=1)

    plt.subplot(212)
    plt.plot(epoch_list, loss_list, lw=1)
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    # plt.ylim(0, 0.2)
    plt.pause(0.01)


dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)