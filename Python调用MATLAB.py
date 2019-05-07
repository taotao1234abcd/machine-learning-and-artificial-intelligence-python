

# 工程需要使用本地环境的 Python，虚拟环境的 Python 不行
#
# 将包含 Python 解释器的文件夹添加到您的 MATLAB 的路径（如果尚未在路径中）

# 在 MATLAB 命令提示符下
# cd (fullfile(matlabroot,'extern','engines','python'))
# system('python setup.py install')

import time

import matlab
import matlab.engine

import numpy as np

# time_start = time.time()
# time_end = time.time()
# print(time_end - time_start)


bus = np.loadtxt('data/bus.txt')
line = np.loadtxt('data/line.txt')


time_start = time.time()

eng = matlab.engine.start_matlab()

time_end = time.time()
print(time_end - time_start)
time_start = time.time()

# y = eng.hanshuming(matlab.double([1,2]))
# out = eng.hanshuming(matlab.double(bus.tolist()),matlab.double(line.tolist()))

out1, out2 = eng.hanshuming(matlab.double(bus.tolist()),matlab.double(line.tolist()),nargout=2)
# eng.plot(matlab.double(out))
out1 = np.asarray(out1)

eng.quit()


time_end = time.time()
print(time_end - time_start)




# 从 Python 调用 MATLAB 函数

# 使用引擎从 Python 调用 MATLAB 的 sqrt 函数
# sqrt import matlab.engine
# eng = matlab.engine.start_matlab()
# ret = eng.sqrt(4.0)
# print(ret)
#
# 2.0
#

#
#
# 将数组放入 MATLAB 工作区

# 在 Python 中创建一个数组, 并将其放入 MATLAB 工作区。
# import matlab.engine
# eng = matlab.engine.start_matlab()
# px = eng.linspace(0.0,6.28,1000)
# # px是一个 matlab 数组, 但eng.linspace其返回给 python。
# # 要在 MATLAB 中使用它, 请将阵列放入 MATLAB 工作区。
# eng.workspace['mx'] = px
# 将条目添加到引擎 workspace 字典时, 也会创建一个 matlab 变量。
# 引擎将数据转换为 MATLAB 数据类型。
#
#
#
# 从 MATLAB 工作区获取数据 

# 从 MATLAB 工作区获取pi , 并将其复制到 python 变量。
# import matlab.engine
# eng = matlab.engine.start_matlab()
# eng.eval('a = pi;',nargout=0)
# mpi = eng.workspace['a']
# print(mpi)
#
# 3.14159265359

