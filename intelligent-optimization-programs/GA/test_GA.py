#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gaft import GAEngine
from gaft.components import BinaryIndividual
from gaft.components import Population
from gaft.operators import TournamentSelection
from gaft.operators import UniformCrossover
from gaft.operators import FlipBitBigMutation

# Built-in best fitness analysis.
from gaft.analysis.fitness_store import FitnessStore
from gaft.analysis.console_output import ConsoleOutput

from time import time
import numpy as np
from numpy import flatnonzero as find
from PyPST.loadflow import loadflow
from PyPST.loadflow_dc import loadflow_dc
from PyPST.loadflow_dc_pro import loadflow_dc_pro

# ---------------------------------------------------------------------------------------------------

bus = np.loadtxt('data/northeast/bus.txt')
line = np.loadtxt('data/northeast/line.txt')

# bus[:, 5] *= 1.05
# bus[:, 6] *= 1.05

pv = find(bus[:, 9] == 2)

myfun_lower_bound = 0
myfun_upper_bound = 1
bounds = np.zeros([len(pv), 2])
bounds[:, 0] = myfun_lower_bound
bounds[:, 1] = myfun_upper_bound
bounds = bounds.tolist()

generations = 15
population_size = 20

# ---------------------------------------------------------------------------------------------------

# Define population.
indv_template = BinaryIndividual(ranges=bounds, eps=0.001)
population = Population(indv_template=indv_template, size=population_size).init()

# Create genetic operators.
#selection = RouletteWheelSelection()
selection = TournamentSelection()
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitBigMutation(pm=0.1, pbm=0.55, alpha=0.6)

# Create genetic algorithm engine.
# Here we pass all built-in analysis to engine constructor.
engine = GAEngine(population=population, selection=selection,
                  crossover=crossover, mutation=mutation)


# ===================================================================================================


def Theta_abs_mean(dc_bus_sol):
    dc_bus_sol = dc_bus_sol.copy()
    Theta_slack = dc_bus_sol[find(dc_bus_sol[:, 5] == 1)[0], 2]
    dc_bus_sol[:, 2] -= Theta_slack
    dc_ThetaMean = np.mean(abs(dc_bus_sol[:, 2]))

    return dc_ThetaMean


# def myfun(x):
#     bus[pv, 15] = np.rint(x)
#     # bus[pv, 15] = 0
#     dc_bus_sol, _, _ = loadflow_dc_pro(bus, line, printout=False)
#     dc_ThetaMean = Theta_abs_mean(dc_bus_sol)
#     print(dc_ThetaMean)
#     return dc_ThetaMean

# Define fitness function.
@engine.fitness_register
def fitness(indv):
    x = np.array(indv.solution)
    # y = -float(sum(x**2 - 10 * np.cos(2*np.pi*x) + 10))
    bus[pv, 15] = np.rint(x)
    # bus[pv, 15] = 0
    dc_bus_sol, _, _ = loadflow_dc_pro(bus, line, printout=False)
    dc_ThetaMean = Theta_abs_mean(dc_bus_sol)
    print(dc_ThetaMean)
    return -float(dc_ThetaMean)

if '__main__' == __name__:
    time_start = time()
    engine.run(ng=generations)
    time_end = time()
    print("GA costs %.4f seconds!" % (time_end - time_start))
    best_indv = engine.population.best_indv(engine.fitness)
    print(best_indv.solution)
    print(-engine.fitness(best_indv))
    best_population_point = best_indv.solution
    bus[pv, 15] = np.rint(best_population_point)
    converge, bus_sol, line_f_from, line_f_to = loadflow(bus, line, printout=True)

