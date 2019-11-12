
import time
import random
import numpy as np
from numpy import flatnonzero as find
import matplotlib.pyplot as plt

from PyPST.loadflow import loadflow
from PyPST.loadflow_dc import loadflow_dc
from PyPST.loadflow_dc_pro import loadflow_dc_pro

# ---------------------------------------------------------------------------------------------------

# def myfun(x):
#     return sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)

# ---------------------------------------------------------------------------------------------------

def myfun(x):
    bus[pv, 15] = np.rint(x)
    # bus[pv, 15] = 0
    dc_bus_sol, _, _ = loadflow_dc_pro(bus, line, printout=False)
    dc_ThetaMean = Theta_abs_mean(dc_bus_sol)
    print(dc_ThetaMean)
    return dc_ThetaMean

def Theta_abs_mean(dc_bus_sol):
    dc_bus_sol = dc_bus_sol.copy()
    Theta_slack = dc_bus_sol[find(dc_bus_sol[:, 5] == 1)[0], 2]
    dc_bus_sol[:, 2] -= Theta_slack
    dc_ThetaMean = np.mean(abs(dc_bus_sol[:, 2]))
    return dc_ThetaMean

bus = np.loadtxt('data/northeast/bus.txt')
line = np.loadtxt('data/northeast/line.txt')

# bus[:, 5] *= 1.05
# bus[:, 6] *= 1.05

pv = find(bus[:, 9] == 2)

# 差分进化算法程序

time_start = time.time()

generation_max = 30  # 最大迭代次数
population_size = 20  # 种群中的个体数
F0 = 0.5  # 变异率
CR = 0.3  # 交叉概率
myfun_variables_num = len(pv)  # 所求问题的维数
Gmin = np.zeros(generation_max)  # 各代的最优值
best_x = np.zeros([generation_max, myfun_variables_num])  # 各代的最优解
value = np.zeros(population_size)
trace = np.zeros([generation_max, 2])  # 储存各代最优的函数值结果

# 产生初始种群
myfun_lower_bound = 0
myfun_upper_bound = 1

# 产生 population_size 个 myfun_variables_num 维的向量
XG = (myfun_upper_bound - myfun_lower_bound) * np.random.rand(population_size, myfun_variables_num) + myfun_lower_bound

value_XG = np.zeros(population_size)
for i in range(population_size):
    value_XG[i] = myfun(XG[i, :])

XG_next = np.zeros([population_size, myfun_variables_num])

G = 1
while G <= generation_max:
    print("-" * 20, "generation %2d" % G, "-" * 20)

    for i in range(population_size):

        # ------------------------      变异操作       ---------------------------

        # 产生 j, k, p 三个不同的索引
        a = 1
        b = population_size
        dx = random.sample(range(b), b)
        j = dx[0]
        k = dx[1]
        p = dx[2]

        # 要保证 j, k, p 与 i 不同
        if j == i:
            j = dx[3]
        if k == i:
            k = dx[3]
        if p == i:
            p = dx[3]

        # 变异算子
        F = F0 * 2 ** np.exp(1 - generation_max / (generation_max + 1 - G))

        # 变异的个体来自 3 个随机父代
        son = XG[p, :] + F * (XG[j, :] - XG[k, :])
        for j in range(myfun_variables_num):
            if son[j] > myfun_lower_bound and son[j] < myfun_upper_bound:  # 变异超出边界, 则重新随机生成
                XG_next[i, j] = son[j]
            else:
                XG_next[i, j] = (myfun_upper_bound - myfun_lower_bound) * np.random.rand(1)[0] + myfun_lower_bound

    # ------------------------      交叉操作       ---------------------------

    for i in range(population_size):
        randx = random.sample(range(myfun_variables_num), myfun_variables_num)
        for j in range(myfun_variables_num):
            if np.random.rand() > CR and randx[0] != j:
                XG_next[i, j] = XG[i, j]

    # ------------------------      选择操作       ---------------------------

    value_XG_next = np.zeros(population_size)
    for i in range(population_size):
        value_XG_next[i] = myfun(XG_next[i, :])

    for i in range(population_size):
        if value_XG_next[i] > value_XG[i]:
            XG_next[i, :] = XG[i, :]
            value_XG_next[i] = value_XG[i]

    # 找出最小值
    for i in range(population_size):
        value[i] = value_XG_next[i]

    value_min = min(value)
    pos_min = find(value == value_min)[0]

    # 第 G 代中的目标函数的最小值
    Gmin[G - 1] = value_min

    # 保存最优的个体
    best_x[G - 1, :] = XG_next[pos_min, :]

    XG = XG_next.copy()
    value_XG = value_XG_next.copy()
    trace[G - 1, 0] = G
    trace[G - 1, 1] = value_min
    G += 1


value_min = min(Gmin)
pos_min = find(Gmin == value_min)[0]
best_population_target = value_min
best_population_point = best_x[pos_min, :].copy()
time_end = time.time()
print("DE costs %.4f seconds!" % (time_end - time_start))

plt.plot(trace[:, 0], trace[:, 1])
plt.show()

print(best_population_point)
print(best_population_target)


bus[pv, 15] = np.rint(best_population_point)
converge, bus_sol, line_f_from, line_f_to = loadflow(bus, line, printout=True)