
import numpy as np
from numpy import flatnonzero as find

# ==========================================================================================================
# 读取某个 PSASP LF数据文件的数据

def read_runPath_LFs(fileName):
    list_Lx = []
    with open(fileName) as f:
        tt = f.readlines()
        for i in tt:
            list_Lx.append(list(eval(i.strip())))
    return list_Lx

# ==========================================================================================================

list_L2 = read_runPath_LFs(fileName="LF.L2")  # 交流线、串联电容电抗器、并联电容电抗器数据 LF.L2
list_L3 = read_runPath_LFs(fileName="LF.L3")  # 变压器数据 LF.L3
list_L5 = read_runPath_LFs(fileName="LF.L5")  # 发电机数据 LF.L5
list_L6 = read_runPath_LFs(fileName="LF.L6")  # 负荷数据 LF.L6

list_L2_array = np.zeros(0)
for i in range(len(list_L2)):
    templine = np.array(list_L2[i][0:7])
    if int(templine[0]) == 1:
        if list_L2_array.shape[0] == 0:
            list_L2_array = templine
        else:
            list_L2_array = np.vstack([list_L2_array, templine])
list_L2_array = np.delete(list_L2_array, [0,3], axis=1)
del templine, list_L2

list_L3_array = np.zeros(0)
for i in range(len(list_L3)):
    templine = np.array(list_L3[i][0:17])
    if int(templine[0]) == 1:
        if list_L3_array.shape[0] == 0:
            list_L3_array = templine
        else:
            list_L3_array = np.vstack([list_L3_array, templine])
list_L3_array = np.delete(list_L3_array, [0,3,7,8,9,11,12,13,14,15], axis=1)
del templine, list_L3


list_L5_array = np.zeros(0)
for i in range(len(list_L5)):
    templine = np.array(list_L5[i][0:7])
    if list_L5_array.shape[0] == 0:
        list_L5_array = templine
    else:
        list_L5_array = np.vstack([list_L5_array, templine])
del templine, list_L5
temp = list_L5_array[:, 0:1]
list_L5_array = np.delete(list_L5_array, [0], axis=1)
list_L5_array = np.hstack([list_L5_array, temp])


list_L6_array = np.zeros(0)
for i in range(len(list_L6)):
    templine = np.array(list_L6[i][0:6])
    if int(templine[0]) == 1:
        if list_L6_array.shape[0] == 0:
            list_L6_array = templine
        else:
            list_L6_array = np.vstack([list_L6_array, templine])
del templine, list_L6
list_L6_array = np.delete(list_L6_array, [0, 2, 3], axis=1)



Nodes = np.union1d(list_L2_array[:,0], list_L2_array[:,1])
Nodes = np.union1d(Nodes, list_L3_array[:,0])
Nodes = np.union1d(Nodes, list_L3_array[:,1])

NumNode = Nodes.shape[0]

bus = np.zeros([NumNode,16])
bus[:,0] = Nodes
bus[:,1] = 1
bus[:,9] = 3

bus[:,15] = 1

line = np.zeros(0)
for i in range(list_L2_array.shape[0]):
    templine = np.zeros([1,7])
    if int(list_L2_array[i,0]) != int(list_L2_array[i,1]):
        templine[0, 0:5] = list_L2_array[i,0:5]
        templine[0, 5] = 1
        templine[0, 4] *= 2
        if line.shape[0] == 0:
            line = templine
        else:
            line = np.vstack([line, templine])
    else:
        tempZ = list_L2_array[i,2] + 1j * list_L2_array[i,3]
        tempY = 1/tempZ
        tempIndex = find(bus[:,0]==list_L2_array[i,0])[0]
        bus[tempIndex, 7] = tempY.real
        bus[tempIndex, 8] = tempY.imag
try:
    del tempZ, tempY
except:
    pass

for i in range(list_L3_array.shape[0]):
    templine = np.zeros([1, 7])
    templine[0, 0:4] = list_L3_array[i, 0:4]
    templine[0, 5] = list_L3_array[i, 4]
    if list_L3_array[i, 5] == 1:
        templine[0, 6] = list_L3_array[i, 6]
    if line.shape[0] == 0:
        line = templine
    else:
        line = np.vstack([line, templine])

temp = line[:,0].copy()
line[:,0] = line[:,1].copy()
line[:,1] = temp
del temp


for i in range(list_L5_array.shape[0]):
    if find(bus[:, 0] == list_L5_array[i, 0]).shape[0] == 0:
        pass
    else:
        tempIndex = find(bus[:, 0] == list_L5_array[i, 0])[0]
    if int(list_L5_array[i, 1]) == 1:
        pass
    elif int(list_L5_array[i, 1]) == 0:
        bus[tempIndex, 9] = 1
    else:
        bus[tempIndex, 9] = 2
    bus[tempIndex, 1] = list_L5_array[i, 4]
    bus[tempIndex, 2] = list_L5_array[i, 5]
    bus[tempIndex, 3] = list_L5_array[i, 2]
    bus[tempIndex, 4] = list_L5_array[i, 3]

    bus[tempIndex, 15] = list_L5_array[i, 6]


for i in range(list_L6_array.shape[0]):
    tempIndex = find(bus[:, 0] == list_L6_array[i, 0])[0]
    bus[tempIndex, 5] = list_L6_array[i, 1]
    bus[tempIndex, 6] = list_L6_array[i, 2]

del i, templine, tempIndex


np.savetxt("bus.txt", bus)
np.savetxt("line.txt", line)


