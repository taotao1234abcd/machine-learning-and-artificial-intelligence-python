#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: test_PSO.py
@time: 2018/06/08 23:31
@description:
test for PSO
"""
# import sys
# sys.path.append("..")

import numpy as np
from numpy import flatnonzero as find
from sopt.util.ga_config import ga_config
import math
from functools import reduce

from time import time
from sopt.util.pso_config import *
from sopt.PSO.PSO import PSO
from sopt.util.constraints import *

from PyPST.loadflow import loadflow
from PyPST.loadflow_dc import loadflow_dc
from PyPST.loadflow_dc_pro import loadflow_dc_pro



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


myfun_lower_bound = 0
myfun_upper_bound = 1
myfun_variables_num = len(pv)
myfun_func_type = ga_config.func_type_min


def myfun(x):
    bus[pv, 15] = np.rint(x)
    # bus[pv, 15] = 0
    dc_bus_sol, _, _ = loadflow_dc_pro(bus, line, printout=False)
    dc_ThetaMean = Theta_abs_mean(dc_bus_sol)
    print(dc_ThetaMean)
    return dc_ThetaMean
    # return sum((x - x**2)**2)


# myfun_lower_bound = -8
# myfun_upper_bound = 8
# myfun_variables_num = 2
# myfun_func_type = ga_config.func_type_min
#
#
# def myfun(x):
#     print(sum(x**2 - 10 * np.cos(2*np.pi*x) + 10))
#     return sum(x**2 - 10 * np.cos(2*np.pi*x) + 10)


# ===================================================================================================


class TestPSO:
    def __init__(self):
        self.func = myfun
        self.func_type = myfun_func_type
        self.variables_num = myfun_variables_num
        self.lower_bound = myfun_lower_bound
        self.upper_bound = myfun_upper_bound
        self.c1 = pso_config.c1
        self.c2 = pso_config.c2
        self.generations = 15
        self.population_size = 20
        self.vmax = 0.4
        self.vmin = -self.vmax
        self.w_start = 0.9
        self.w_end = 0.4
        self.w_method = pso_w_method.linear_decrease
        #self.complex_constraints = [constraints1,constraints2,constraints3]
        self.complex_constraints = None
        self.complex_constraints_method = complex_constraints_method.loop
        self.PSO = PSO(**self.__dict__)

    def test(self):
        start_time = time()
        self.PSO.run()
        print("PSO costs %.4f seconds!" %(time()-start_time))
        self.PSO.save_plot()
        self.PSO.show_result()
        return np.array(self.PSO.generations_best_points[self.PSO.global_best_index]), \
               np.array(self.PSO.generations_best_targets[self.PSO.global_best_index])


if __name__ == '__main__':
    best_population_point, best_population_target = TestPSO().test()
    bus[pv, 15] = np.rint(best_population_point)
    converge, bus_sol, line_f_from, line_f_to = loadflow(bus, line, printout=True)
