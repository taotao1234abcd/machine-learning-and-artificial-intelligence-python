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
import os
import numpy as np

# Define population.
indv_template = BinaryIndividual(ranges=[(-2, 2),
                                         (-2, 2),
                                         (-2, 2)], eps=0.0001)
population = Population(indv_template=indv_template, size=100).init()

# Create genetic operators.
#selection = RouletteWheelSelection()
selection = TournamentSelection()
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitBigMutation(pm=0.1, pbm=0.55, alpha=0.6)

# Create genetic algorithm engine.
# Here we pass all built-in analysis to engine constructor.
engine = GAEngine(population=population, selection=selection,
                  crossover=crossover, mutation=mutation,
                  analysis=[ConsoleOutput, FitnessStore])

# Define fitness function.
@engine.fitness_register
def fitness(indv):
    x = np.zeros([3,1])
    x = indv.solution
    return -((x[0] - 1)**2 + (x[1] - 1.5)**2 + (x[2] + 0.5)**2)

if '__main__' == __name__:
    engine.run(ng=50)

    os.remove('best_fit.py')

    best_indv = engine.population.best_indv(engine.fitness)
    print(best_indv.solution)
    print(engine.fitness(best_indv))
