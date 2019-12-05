
import sys, os

# 禁止屏幕打印
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# 允许屏幕打印
def enablePrint():
    sys.stdout = sys.__stdout__


print('打印1')
blockPrint()
print('打印2')
enablePrint()
print('打印3')

