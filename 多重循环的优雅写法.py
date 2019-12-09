
# product 返回一个生成器，能 yield 出传入数组形成的笛卡尔积
# product 可接受任意多个数组

from itertools import product
lit = [1, 2, 3]
for i, j, k in product(lit, lit, lit):
    print(i, j, k)
