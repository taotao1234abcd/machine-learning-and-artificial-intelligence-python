
import numpy as np
import torch


torch.manual_seed(0)
torch.cuda.manual_seed(0)
print(torch.__version__)


x = torch.tensor([5.5, 3])
print(x)

x = torch.arange(2, 5)
print(x)

x = torch.rand(5, 3)
print(x)


np_data = np.arange(6).reshape(2, 3)
# 使用 view() 来改变 Tensor 的形状：
y = x.view(15)
z = x.view(-1, 5)  # -1所指的维度可以根据其他维度的值推出来
print(y)
print(z)
# 注意 view() 返回的新 tensor 与源 tensor 共享内存，改变其中一个时另一个也会改变
# 如果不想共享内存，应先用 clone 创造一个副本。
x_cp = x.clone()


# Tensor 和 NumPy 相互转换
a = torch.ones(5)
b = a.numpy()
print(a, b)
# numpy() 和 from_numpy() 这两个函数产生的 Tensor 和 NumPy 数组共享内存，改变其中一个时另一个也会改变
a = np.ones(5)
b = torch.from_numpy(a)
print(a, b)

# 直接用 torch.tensor() 将 NumPy 数组转换成 Tensor 会进行数据拷贝，不共享内存
c = torch.tensor(a)
print(a, c)


if torch.cuda.is_available():
    device = torch.device("cuda:0")          # GPU
    y = torch.ones_like(x, device=device)  # 直接创建一个在 GPU 上的 Tensor
    x = x.to(device)                       # 等价于 .to("cuda:0")
    z = x + y
    print(z)
    print(z.to("cpu", dtype=torch.float64))       # to() 还可以同时更改数据类型


# abs
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)  # 32-bit floating point
print(
    '\nabs',
    '\nnumpy: ', np.abs(data),          # [1 2 1 2]
    '\ntorch: ', torch.abs(tensor)      # [1 2 1 2]
)

# sin
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)  # 32-bit floating point
print(
    '\nsin',
    '\nnumpy: ', np.sin(data),      # [-0.84147098 -0.90929743  0.84147098  0.90929743]
    '\ntorch: ', torch.sin(tensor)  # [-0.8415 -0.9093  0.8415  0.9093]
)

# mean
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)  # 32-bit floating point
print(
    '\nmean',
    '\nnumpy: ', np.mean(data),         # 0.0
    '\ntorch: ', torch.mean(tensor)     # 0.0
)

# matrix multiplication
data = [[1,2], [3,4]]
tensor = torch.FloatTensor(data)  # 32-bit floating point
# correct method
print(
    '\nmatrix multiplication (matmul)',
    '\nnumpy: ', np.matmul(data, data),     # [[7, 10], [15, 22]]
    '\ntorch: ', torch.mm(tensor, tensor)   # [[7, 10], [15, 22]]
)
# # incorrect method
# data = np.array(data)
# print(
#     '\nmatrix multiplication (dot)',
#     '\nnumpy: ', data.dot(data),        # [[7, 10], [15, 22]]
#     '\ntorch: ', tensor.dot(tensor)     # RuntimeError: dot: Expected 1-D argument self, but got 2-D
# )

