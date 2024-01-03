import random
import time
from copy import copy

import numpy as np
from icecream import ic
import networkx as nx

class RegularGraph:
    """
    Class implementing a regular graph population.
    """
    def __init__(self, populationSize: int, transientGenNum: int, genNum: int, graphNum: int, runNum: int,
                 initCooperatorsFraction: float,
                 graphConnectivity: int,
                 contributionValue: int,
                 contributionModel: int, evolutionModel: int, updateStrategy: int, mutations: bool):
        """
        :param populationSize:  number of nodes in the graph
        :param transientGenNum:  transient number of generations before taking data
        :param genNum:  number of generations
        :param graphNum: number of instances of graphs of the same class
        :param runNum:  number of runs for each instance of a graph
        :param initCooperatorsFraction:  initial cooperators fraction in the population
        :param graphConnectivity:  graph connectivity
        :param contributionValue:  contribution value
        :param contributionModel: (0: cost per game, 1: cost per individual)
        :param evolutionModel: (0: pairwise comparison, 1: death-birth , 2: birth-death)
        :param updateStrategy: (0: synchronous, 1: asynchronous)
        :param mutations: true if mutations are allowed
        """
        self.populationSize = populationSize
        self.transientGenNum = transientGenNum
        self.genNum = genNum
        self.graphNum = graphNum
        self.runNum = runNum
        self.initCooperatorsFraction = initCooperatorsFraction
        self.graphConnectivity = graphConnectivity
        self.contributionValue = contributionValue
        self.contributionModel = contributionModel
        self.updateStrategy = updateStrategy
        self.evolutionModel = evolutionModel
        self.mutations = mutations

        self.r = 1
        self.populationDict = {}

    def computeFitness(self, population, node):
        fitness = 0
        minPayoff = 0
        maxPayoff = 0
        neighborsIndices = self.populationDict[node]
        neighborsStrat = self.extractIndividuals(population, neighborsIndices)

        if self.contributionModel == 0:  # cost per game
            c = neighborsStrat.count(1) + population[node]
            k = len(self.populationDict[node])

            fitness += self.r * c / (k + 1) - population[node]
            minPayoff += self.r * (population[node]) / (k + 1) - population[node]
            maxPayoff += self.r * (len(neighborsIndices) + population[node]) / (k + 1) - population[node]

            for neighborIndex in neighborsIndices:
                c = self.extractIndividuals(population, self.populationDict[neighborIndex]).count(1) + population[neighborIndex]
                k = len(self.populationDict[neighborIndex])

                fitness += self.r * c / (k + 1) - population[node]
                minPayoff += -population[node]
                maxPayoff += self.r * (len(self.populationDict[neighborIndex]) + 1) / (k + 1) - population[node]

        else:  # cost per individual
            k_y = len(self.populationDict[node])
            fitness += self.r / (k_y+1)
            minPayoff += self.r / (k_y+1)
            maxPayoff += self.r / (k_y+1)

            totalContribution = (1/(k_y+1))*population[node] - (1/(k_y+1))*population[node]
            totalContributionMax = (1/(k_y+1))*population[node] - (1/(k_y+1))*population[node]
            totalContributionMin = (1/(k_y+1))*population[node] - (1/(k_y+1))*population[node]
            for neighborIndex in neighborsIndices:
                k_i = len(self.populationDict[neighborIndex])
                totalContribution += (1/(k_i+1))*population[neighborIndex] - (1/(k_y+1))*population[node]
                totalContributionMax += (1/(k_i+1)) - (1/(k_y+1))*population[node]
                totalContributionMin += -(1/(k_y+1))*population[node]

            fitness = fitness*totalContribution
            maxPayoff = maxPayoff*totalContributionMax
            minPayoff = minPayoff*totalContributionMin

            for neighborIndex in neighborsIndices:
                neighborsIndices2 = self.populationDict[neighborIndex]
                k_x = len(neighborsIndices2)
                fitness2 = self.r / (k_x + 1)
                minPayoff2 = self.r / (k_x + 1)
                maxPayoff2 = self.r / (k_x + 1)

                totalContribution = (1 / (k_x + 1)) * population[neighborIndex] - (1 / (k_y + 1)) * population[node]
                totalContributionMax = (1 / (k_x + 1)) - (1 / (k_y + 1)) * population[node]
                totalContributionMin = -(1 / (k_y + 1)) * population[node]
                for neighborIndex2 in neighborsIndices2:
                    k_i = len(self.populationDict[neighborIndex2])
                    totalContribution += (1 / (k_i + 1)) * population[neighborIndex2] - (1 / (k_y + 1)) * population[node]
                    totalContributionMax += (1 / (k_i + 1)) - (1 / (k_y + 1)) * population[node]
                    totalContributionMin += -(1 / (k_y + 1)) * population[node]

                fitness2 = fitness2 * totalContribution
                maxPayoff2 = maxPayoff2 * totalContributionMax
                minPayoff2 = minPayoff2 * totalContributionMin

                fitness += fitness2
                maxPayoff += maxPayoff2
                minPayoff += minPayoff2

        return fitness, minPayoff, maxPayoff

    def imitationProbability(self, population, node, neighbor):
        nodeFitness, nodeMinPayoff, nodeMaxPayoff = self.computeFitness(population, node)
        neighborFitness, neighborMinPayoff, neighborMaxPayoff = self.computeFitness(population, neighbor)
        if nodeFitness > neighborFitness:
            return -1
        M = max(abs(nodeMaxPayoff-neighborMinPayoff), abs(neighborMaxPayoff-nodeMinPayoff))

        return (neighborFitness-nodeFitness) / M

    def nextGen(self, population):
        if self.evolutionModel == 0:  # pairwise comparison
            for node in self.populationDict:
                neighbor = random.choice(self.populationDict[node])
                if random.random() < self.imitationProbability(population, node, neighbor):
                    population[node] = population[neighbor]

        elif self.evolutionModel == 1:  # death-birth
            pass

        elif self.evolutionModel == 2:  # birth-death
            pass

        return population

    def simulate(self) -> np.array([[int], [int]]):
        """
        :return: [x_values: n=r/(z+1) renormalized PGG enhancement factor, y_values: fraction of cooperators]
        """
        cooperators = np.ones(int(self.populationSize * self.initCooperatorsFraction))
        defectors = np.zeros(int(self.populationSize * (1 - self.initCooperatorsFraction)))
        populationOriginal = np.hstack((cooperators, defectors))

        valuesPerGraphs = np.zeros((self.graphNum, 2, (self.graphConnectivity+1)*5 + - 1))  # number of fractions or r
        valuesPerRuns = np.zeros((self.runNum, 2, (self.graphConnectivity+1)*5 + - 1))  # number of fractions of r
        valuesPerGens = np.zeros(self.genNum)

        #for r in range(21, (self.graphConnectivity+1)*5+4):
        r = 24
        self.r = (r/5)
        ic(r)

        for graph in range(self.graphNum):
            # Create graph representing the population indices (will be used to represent the structure)
            populationGraphIndices = nx.random_regular_graph(self.graphConnectivity, self.populationSize)
            self.populationDict = {}
            for node in populationGraphIndices.nodes():  # for optimality reasons
                self.populationDict[node] = list(populationGraphIndices.neighbors(node))

            for run in range(self.runNum):
                # Create and shuffle population (0: D, 1:C)
                population = copy(populationOriginal)
                random.shuffle(population)
                ic(run)
                for _ in range(self.transientGenNum):
                    self.nextGen(population)

                for gen in range(self.genNum):
                    self.nextGen(population)
                    valuesPerGens[gen] = np.count_nonzero(population) / self.populationSize

                valuesPerRuns[run, 0, r-5] = self.r/(self.graphConnectivity + 1)
                valuesPerRuns[run, 1, r-5] = np.mean(valuesPerGens)

            valuesPerGraphs[graph] = np.mean(valuesPerRuns, axis=0)

        simulationValuesForRegularGraph = np.mean(valuesPerGraphs, axis=0)
        ic(simulationValuesForRegularGraph)
        return simulationValuesForRegularGraph

    def simulateWithWealth(self) -> np.array([[int], [int]]):
        """
        :return: [x_values: their fraction of the total wealth, y_values: number of individuals]
        """
        populationOriginal = np.ones(int(self.populationSize))

        valuesPerGraphs = np.zeros((self.graphNum, 2, (self.graphConnectivity + 1) * 5 + - 1))  # number of fractions or r
        valuesPerGens = np.zeros(self.genNum)

        for graph in range(self.graphNum):
            # Create graph representing the population indices (will be used to represent the structure)
            populationGraphIndices = nx.random_regular_graph(self.graphConnectivity, self.populationSize)
            self.populationDict = {}
            for node in populationGraphIndices.nodes():  # for optimality reasons
                self.populationDict[node] = list(populationGraphIndices.neighbors(node))

            # Create and shuffle population (0: D, 1:C)
            population = copy(populationOriginal)
            random.shuffle(population)

            for _ in range(self.transientGenNum):
                self.nextGen(population)

            for gen in range(self.genNum):
                self.nextGen(population)
                valuesPerGens[gen] = np.count_nonzero(population) / self.populationSize

            valuesPerGraphs[graph, 0] = self.r / (self.graphConnectivity + 1)
            valuesPerGraphs[graph, 1] = np.mean(valuesPerGens)

        simulationValuesForRegularGraphWithWealth = np.mean(valuesPerGraphs, axis=0)
        ic(simulationValuesForRegularGraphWithWealth)
        return simulationValuesForRegularGraphWithWealth


    def extractIndividuals(self, population, indices):
        return [population[i] for i in indices]






