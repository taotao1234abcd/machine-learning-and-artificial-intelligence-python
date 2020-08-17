

# 工程需要使用本地环境的 Python，虚拟环境的 Python 不行
#
# 将包含 Python 解释器的文件夹添加到您的 MATLAB 的路径（如果尚未在路径中）
# MATLAB R2018b 需要搭配 Python 3.6 使用，具体版本匹配要求见 matlabroot\extern\engines\python\setup.py 中所写

# 在 MATLAB 命令提示符下
# cd (fullfile(matlabroot,'extern','engines','python'))
# system('python setup.py install')

# 默认情况下，安装程序将在 matlabroot\extern\engines\python 文件夹编译用于 Python® 的引擎 API。安装程序将引擎安装在默认的 Python 文件夹中。如果您没有这些文件夹的写入权限，请选择以下非默认选项之一。如果安装在另一个文件夹中，请将 PYTHONPATH 变量更新为该文件夹的位置。
# 
# 下面是编译和安装引擎 API 的选项以及在操作系统提示符下输入的命令。
# 
# 在非默认文件夹中编译，在默认文件夹中安装
# 如果您不具备在 MATLAB® 文件夹中编译引擎的写入权限，请使用非默认文件夹 builddir。
# 
# cd "matlabroot\extern\engines\python"
# python setup.py build --build-base="builddir" install
# 在默认文件夹中编译，在非默认文件夹中安装
# 如果您不具备在默认的 Python 文件夹中安装引擎的写入权限，请使用非默认文件夹 installdir。
# 
# cd "matlabroot\extern\engines\python"
# python setup.py install --prefix="installdir"
# 要在 Python 包的搜索路径中包含 installdir，请将 installdir 添加到 PYTHONPATH 环境变量。
# 
# 在非默认文件夹中编译和安装
# 如果您对 MATLAB 文件夹和默认的 Python 文件夹都没有写入权限，则可以指定非默认文件夹。对于编译文件夹，使用 builddir，对于安装文件夹，使用 installdir。
# 
# cd "matlabroot\extern\engines\python"
# python setup.py build --build-base="builddir" install --prefix="installdir"

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

# y = eng.函数名(matlab.double([1,2]))
# out = eng.函数名(matlab.double(bus.tolist()),matlab.double(line.tolist()))

out1, out2 = eng.函数名(matlab.double(bus.tolist()),matlab.double(line.tolist()),nargout=2)
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

