
from time import time
import numpy as np
from numpy import flatnonzero as find
import copy
import matplotlib.pyplot as plt


def myfun(x):

    return sum(x**2 - 10 * np.cos(2*np.pi*x) + 10)


class AFSIndividual:
    """class for AFSIndividual"""

    def __init__(self, vardim, bound):
        '''
        vardim: dimension of variables
        bound: boundaries of variables
        '''
        self.vardim = vardim
        self.bound = bound
        self.func = myfun

    def generate(self):
        '''
        generate a rondom chromsome
        '''
        len = self.vardim
        rnd = np.random.random(size=len)
        self.chrom = np.zeros(len)
        self.velocity = np.random.random(size=len)
        for i in range(0, len):
            self.chrom[i] = self.bound[0, i] + \
                            (self.bound[1, i] - self.bound[0, i]) * rnd[i]
        self.bestPosition = np.zeros(len)
        self.bestFitness = 0.

    def calculateFitness(self):
        '''
        calculate the fitness of the chromsome
        '''
        self.fitness = -self.func(self.chrom)


class AFS:
    """class for  ArtificialFishSwarm"""

    def __init__(self, population_size, vardim, bound, generations, params):
        '''
        population_size: population population_size
        vardim: dimension of variables
        bound: boundaries of variables, 2*vardim
        generations: termination condition
        params: algorithm required parameters, it is a list which is consisting of[visual, step, delta, trynum]
        '''
        self.population_size = population_size
        self.vardim = vardim
        self.bound = bound
        self.generations = generations
        self.params = params
        self.population = []
        self.fitness = np.zeros((self.population_size, 1))
        self.trace = np.zeros((self.generations, 1))
        self.lennorm = 6000

    def initialize(self):
        '''
        initialize the population of afs
        '''
        for i in range(0, self.population_size):
            ind = AFSIndividual(self.vardim, self.bound)
            ind.generate()
            self.population.append(ind)

    def evaluation(self, x):
        '''
        evaluation the fitness of the individual
        '''
        x.calculateFitness()

    def forage(self, x):
        '''
        artificial fish foraging behavior
        '''
        newInd = copy.deepcopy(x)
        found = False
        for i in range(0, self.params[3]):
            indi = self.randSearch(x, self.params[0])
            if indi.fitness > x.fitness:
                newInd.chrom = x.chrom + np.random.random(self.vardim) * self.params[1] * self.lennorm * (
                        indi.chrom - x.chrom) / np.linalg.norm(indi.chrom - x.chrom)
                newInd = indi
                found = True
                break
        if not found:
            newInd = self.randSearch(x, self.params[1])
        return newInd

    def randSearch(self, x, searLen):
        '''
        artificial fish random search behavior
        '''
        ind = copy.deepcopy(x)
        ind.chrom += np.random.uniform(-1, 1,
                                       self.vardim) * searLen * self.lennorm
        for j in range(0, self.vardim):
            if ind.chrom[j] < self.bound[0, j]:
                ind.chrom[j] = self.bound[0, j]
            if ind.chrom[j] > self.bound[1, j]:
                ind.chrom[j] = self.bound[1, j]
        self.evaluation(ind)
        return ind

    def huddle(self, x):
        '''
        artificial fish huddling behavior
        '''
        newInd = copy.deepcopy(x)
        dist = self.distance(x)
        index = []
        for i in range(1, self.population_size):
            if dist[i] > 0 and dist[i] < self.params[0] * self.lennorm:
                index.append(i)
        nf = len(index)
        if nf > 0:
            xc = np.zeros(self.vardim)
            for i in range(0, nf):
                xc += self.population[index[i]].chrom
            xc = xc / nf
            cind = AFSIndividual(self.vardim, self.bound)
            cind.chrom = xc
            cind.calculateFitness()
            if (cind.fitness / nf) > (self.params[2] * x.fitness):
                xnext = x.chrom + np.random.random(
                    self.vardim) * self.params[1] * self.lennorm * (xc - x.chrom) / np.linalg.norm(xc - x.chrom)
                for j in range(0, self.vardim):
                    if xnext[j] < self.bound[0, j]:
                        xnext[j] = self.bound[0, j]
                    if xnext[j] > self.bound[1, j]:
                        xnext[j] = self.bound[1, j]
                newInd.chrom = xnext
                self.evaluation(newInd)
                # print "hudding"
                return newInd
            else:
                return self.forage(x)
        else:
            return self.forage(x)

    def follow(self, x):
        '''
        artificial fish following behivior
        '''
        newInd = copy.deepcopy(x)
        dist = self.distance(x)
        index = []
        for i in range(1, self.population_size):
            if dist[i] > 0 and dist[i] < self.params[0] * self.lennorm:
                index.append(i)
        nf = len(index)
        if nf > 0:
            best = -999999999.
            bestIndex = 0
            for i in range(0, nf):
                if self.population[index[i]].fitness > best:
                    best = self.population[index[i]].fitness
                    bestIndex = index[i]
            if (self.population[bestIndex].fitness / nf) > (self.params[2] * x.fitness):
                xnext = x.chrom + np.random.random(
                    self.vardim) * self.params[1] * self.lennorm * (
                                    self.population[bestIndex].chrom - x.chrom) / np.linalg.norm(
                    self.population[bestIndex].chrom - x.chrom)
                for j in range(0, self.vardim):
                    if xnext[j] < self.bound[0, j]:
                        xnext[j] = self.bound[0, j]
                    if xnext[j] > self.bound[1, j]:
                        xnext[j] = self.bound[1, j]
                newInd.chrom = xnext
                self.evaluation(newInd)
                # print "follow"
                return newInd
            else:
                return self.forage(x)
        else:
            return self.forage(x)

    def solve(self):
        '''
        evolution process for afs algorithm
        '''
        self.t = 0
        self.initialize()
        # evaluation the population
        for i in range(0, self.population_size):
            self.evaluation(self.population[i])
            self.fitness[i] = self.population[i].fitness
        best = np.max(self.fitness)
        bestIndex = np.argmax(self.fitness)
        self.best = copy.deepcopy(self.population[bestIndex])
        self.trace[self.t, 0] = self.best.fitness
        print("Generation %d: optimal function value is: %f" % (
            self.t, -self.trace[self.t, 0]))
        while self.t < self.generations - 1:
            self.t += 1
            # newpop = []
            for i in range(0, self.population_size):
                xi1 = self.huddle(self.population[i])
                xi2 = self.follow(self.population[i])
                if xi1.fitness > xi2.fitness:
                    self.population[i] = xi1
                    self.fitness[i] = xi1.fitness
                else:
                    self.population[i] = xi2
                    self.fitness[i] = xi2.fitness
            best = np.max(self.fitness)
            bestIndex = np.argmax(self.fitness)
            if best > self.best.fitness:
                self.best = copy.deepcopy(self.population[bestIndex])
            self.trace[self.t, 0] = self.best.fitness
            print("Generation %d: optimal function value is: %f" % (
                self.t, -self.trace[self.t, 0]))

        print("Optimal function value is: %f; " % -self.trace[self.t, 0])
        print("Optimal solution is:")
        print(self.best.chrom)
        # self.printResult()

    def distance(self, x):
        '''
        return the distance array to a individual
        '''
        dist = np.zeros(self.population_size)
        for i in range(0, self.population_size):
            dist[i] = np.linalg.norm(x.chrom - self.population[i].chrom) / 6000
        return dist

    def printResult(self):
        '''
        plot the result of afs algorithm
        '''
        x = np.arange(0, self.generations)
        y = -self.trace[:, 0]
        plt.plot(x, y, label='optimal value')
        plt.xlabel("Iteration")
        plt.ylabel("function value")
        plt.title("Artificial Fish Swarm algorithm for function optimization")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    time_start = time()

    generations = 50
    population_size = 50

    vardim = 10
    bound = np.tile([[0], [1]], vardim)

    params = [0.01, 0.001, 0.618, 50]

    afs = AFS(population_size, vardim, bound, generations, params)
    afs.solve()

    time_end = time()
    print("AFS costs %.4f seconds!" % (time_end - time_start))


    trace_to_plot = afs.trace
    temp = np.zeros([len(trace_to_plot), 2])
    temp[:, 0] = np.arange(1, len(trace_to_plot) + 1)
    temp[:, 1] = -trace_to_plot[:, 0].copy()
    trace_to_plot = temp.copy()
    del temp
    plt.plot(trace_to_plot[:, 0], trace_to_plot[:, 1])
    plt.show()
