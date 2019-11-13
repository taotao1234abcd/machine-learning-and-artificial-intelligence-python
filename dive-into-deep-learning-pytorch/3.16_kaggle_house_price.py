
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys



def get_k_fold_data(i, x, y, k=5):
    # 返回第i折交叉验证时所需要的训练和验证数据
    assert k > 1
    fold_size = x.shape[0] // k
    x_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        x_part, y_part = x[idx, :], y[idx]
        if j == i:
            x_valid, y_valid = x_part, y_part
        elif x_train is None:
            x_train, y_train = x_part, y_part
        else:
            x_train = torch.cat((x_train, x_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return x_train, y_train, x_valid, y_valid


def k_fold(x_train, y_train, num_epochs, learning_rate, weight_decay, batch_size, k=5):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(i, x_train, y_train)
        train_list, valid_list, _ = train(*data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_list[-1]
        valid_l_sum += valid_list[-1]
        print('fold %d, train rmse %f, valid rmse %f' % (i, train_list[-1], valid_list[-1]))
    return train_l_sum / k, valid_l_sum / k

def log_rmse(net, features, labels):
    with torch.no_grad():
        # # 将小于1的值设成1，使得取对数时数值更稳定
        # clipped_preds = torch.max(net(features), torch.tensor(1.0).cuda())
        rmse = torch.sqrt(((net(features).log() - labels.log())**2).mean())
    return rmse.item()



train_data = pd.read_csv('data/kaggle_house_price/train.csv')
test_data = pd.read_csv('data/kaggle_house_price/test.csv')

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))  # 将训练数据集和测试数据集的特征列矩阵上下拼接

numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index  # 原始数据为数值的特征的索引

all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))  # 数值数据归一化
all_features[numeric_features] = all_features[numeric_features].fillna(0)  # 数值数据的缺失值置为0

all_features = pd.get_dummies(all_features, dummy_na=True)  # dummy_na=True 则将缺失值也视为合法的特征值并为其创建特征
# 将离散的非数值数据转换为数值特征。例如，假设特征 MSZoning 中有两个不同的离散值 RL 和 RM，那么这一步转换将去掉 MSZoning 特征，
# 并新增加两个特征 MSZoning_RL 和 MSZoning_RM，值为 0 或 1。如果一个样本原先在 MSZoning 中的值为 RL, 那么 MSZoning_RL=1, MSZoning_RM=0。

n_train = train_data.shape[0]
num_features = all_features.shape[1]
train_x = torch.tensor(all_features[:n_train].values, dtype=torch.float).cuda()
test_x = torch.tensor(all_features[n_train:].values, dtype=torch.float).cuda()
train_y = torch.tensor(train_data.iloc[:, -1].values, dtype=torch.float).view(-1, 1).cuda()


def train(train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size):

    # 将训练数据的特征和标签组合
    dataset = Data.TensorDataset(train_features, train_labels)

    # 把 dataset 放入 DataLoader
    data_iter = Data.DataLoader(
        dataset=dataset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据, 注意多线程需要在 if __name__ == '__main__': 函数中运行
    )
    # num_workers=0 表示不用额外的进程来加速读取数据

    num_hidden = 200
    net = nn.Sequential(
        nn.Linear(num_features, num_hidden),
        # nn.BatchNorm1d(num_hidden),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(num_hidden, 1)
    ).cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay = weight_decay)
    loss_func = torch.nn.MSELoss()

    train_list = []
    test_list = []
    for epoch in range(num_epochs):
        for step, (b_x, b_y) in enumerate(data_iter):
            prediction = net(b_x)     # input x and predict based on x
            loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

        net.eval()  # 评估模式, 这会关闭dropout
        train_ls = log_rmse(net, train_x, train_y)
        train_list.append(train_ls)
        # print('Epoch: %3d' % (epoch + 1), '| train loss: %.4f' % train_ls)
        try:
            test_ls = log_rmse(net, test_features, test_labels)
            test_list.append(test_ls)
        except:
            pass
        net.train()

        plt.plot(train_list, lw=1)
        plt.plot(test_list, lw=1)
        plt.xlabel('Epoches')
        plt.ylabel('Loss')
        plt.pause(0.01)

    return train_list, test_list, net



def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    train_ls, _, net = train(train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
    print('train rmse %f' % train_ls[-1])
    preds = net(test_features).detach().cpu().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('./submission.csv', index=False)



num_epochs = 20
lr = 0.1
weight_decay = 5
batch_size = 64

train_l, valid_l = k_fold(train_x, train_y, num_epochs, lr, weight_decay, batch_size)
print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (5, train_l, valid_l))

# train_and_pred(train_x, test_x, train_y, test_data, num_epochs, lr, weight_decay, batch_size)