

import os
import xlwt
import xlrd
import copy
import numpy as np

datas = os.listdir('D:/photo1')
datas_s = np.array(datas)
#  C:\Users\Sili_\Desktop\0827研部照片\201
# C:\Users\Sili_\Desktop\photo1
excel_xuhao = xlrd.open_workbook(r'D:/photo1/photo2.xlsx')
sheet_xuhao = excel_xuhao.sheet_by_index(0)
# np.array(a)
rows = sheet_xuhao.nrows
for data in datas_s:
   # name = np.char.split(data, sep='.')[0]
    # p.char.split('www.runoob.com', sep='.')
    name = data
    print(name)
    for row in range(rows):
        if name == sheet_xuhao.cell(row, 1).value:
            xuhao = sheet_xuhao.cell(row, 2).value
            new_data = str(xuhao) + "+" + data
            path_data = 'D:/photo1/' + data
            path_new_data = 'D:/photo1/' + new_data
            try:
                os.rename(path_data, path_new_data)
            except:
                pass
           # del datas[0]
        #else:
           # name = "wu"+name

