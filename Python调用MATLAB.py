

# ������Ҫʹ�ñ��ػ����� Python�����⻷���� Python ����
#
# ������ Python ���������ļ�����ӵ����� MATLAB ��·���������δ��·���У�

# �� MATLAB ������ʾ����
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




# �� Python ���� MATLAB ����

# ʹ������� Python ���� MATLAB �� sqrt ����
# sqrt import matlab.engine
# eng = matlab.engine.start_matlab()
# ret = eng.sqrt(4.0)
# print(ret)
#
# 2.0
#

#
#
# ��������� MATLAB ������

# �� Python �д���һ������, ��������� MATLAB ��������
# import matlab.engine
# eng = matlab.engine.start_matlab()
# px = eng.linspace(0.0,6.28,1000)
# # px��һ�� matlab ����, ��eng.linspace�䷵�ظ� python��
# # Ҫ�� MATLAB ��ʹ����, �뽫���з��� MATLAB ��������
# eng.workspace['mx'] = px
# ����Ŀ��ӵ����� workspace �ֵ�ʱ, Ҳ�ᴴ��һ�� matlab ������
# ���潫����ת��Ϊ MATLAB �������͡�
#
#
#
# �� MATLAB ��������ȡ���� 

# �� MATLAB ��������ȡpi , �����临�Ƶ� python ������
# import matlab.engine
# eng = matlab.engine.start_matlab()
# eng.eval('a = pi;',nargout=0)
# mpi = eng.workspace['a']
# print(mpi)
#
# 3.14159265359

