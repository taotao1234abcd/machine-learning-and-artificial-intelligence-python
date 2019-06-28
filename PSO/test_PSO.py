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
from time import time
from functions import *
from sopt.util.pso_config import *
from sopt.PSO.PSO import PSO
from sopt.util.constraints import *

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
        self.vmax = 1
        self.vmin = -1
        self.w = 1
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
