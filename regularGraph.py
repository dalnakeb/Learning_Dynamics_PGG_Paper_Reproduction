import random
import time
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

    def computeFitness(self, population, populationGraphIndices, node):
        fitness = 0
        minPayoff = 0
        maxPayoff = 0
        neighborsIndices = list(populationGraphIndices.neighbors(node))
        neighbors = self.extractIndividuals(population, neighborsIndices)

        if self.contributionModel == 0:  # cost per game
            c = np.count_nonzero(neighbors) + population[node]
            k = populationGraphIndices.degree[node]

            fitness += self.r * c / (k + 1) - population[node]
            minPayoff += self.r * (population[node]) / (k + 1) - population[node]
            maxPayoff += self.r * (populationGraphIndices.degree[node] + population[node]) / (k + 1) - population[node]

            for neighborIndex in neighborsIndices:
                c = np.count_nonzero(self.extractIndividuals(population, list(populationGraphIndices.neighbors(neighborIndex)))) + population[neighborIndex]
                k = populationGraphIndices.degree[neighborIndex]

                fitness += self.r * c / (k + 1) - population[node]
                minPayoff += self.r * (population[neighborIndex]) / (k + 1) - population[node]
                maxPayoff += self.r * (populationGraphIndices.degree[neighborIndex] + population[neighborIndex]) / (k + 1) - population[node]

        else:  # cost per individual
            k_y = populationGraphIndices.degree[node]
            fitness += self.r / (k_y+1)
            minPayoff += self.r / (k_y+1)
            maxPayoff += self.r / (k_y+1)

            totalContribution = (1/(k_y+1))*population[node] - (1/(k_y+1))*population[node]
            totalContributionMax = (1/(k_y+1))*population[node] - (1/(k_y+1))*population[node]
            totalContributionMin = (1/(k_y+1))*population[node] - (1/(k_y+1))*population[node]
            for neighborIndex in neighborsIndices:
                k_i = populationGraphIndices.degree[neighborIndex]
                totalContribution += (1/(k_i+1))*population[neighborIndex] - (1/(k_y+1))*population[node]
                totalContributionMax += (1/(k_i+1)) - (1/(k_y+1))*population[node]
                totalContributionMin += -(1/(k_y+1))*population[node]

            fitness = fitness*totalContribution
            maxPayoff = maxPayoff*totalContributionMax
            minPayoff = minPayoff*totalContributionMin

            for neighborIndex in neighborsIndices:
                neighborsIndices2 = list(populationGraphIndices.neighbors(neighborIndex))
                k_x = populationGraphIndices.degree[neighborIndex]
                fitness2 = self.r / (k_x + 1)
                minPayoff2 = self.r / (k_x + 1)
                maxPayoff2 = self.r / (k_x + 1)

                totalContribution = (1 / (k_x + 1)) * population[neighborIndex] - (1 / (k_y + 1)) * population[node]
                totalContributionMax = (1 / (k_x + 1)) * population[neighborIndex] - (1 / (k_y + 1)) * population[node]
                totalContributionMin = (1 / (k_x + 1)) * population[neighborIndex] - (1 / (k_y + 1)) * population[node]
                for neighborIndex2 in neighborsIndices2:
                    k_i = populationGraphIndices.degree[neighborIndex2]
                    totalContribution += (1 / (k_i + 1)) * population[neighborIndex2] - (1 / (k_y + 1)) * population[
                        node]
                    totalContributionMax += (1 / (k_i + 1)) - (1 / (k_y + 1)) * population[node]
                    totalContributionMin += -(1 / (k_y + 1)) * population[node]

                fitness2 = fitness2 * totalContribution
                maxPayoff2 = maxPayoff2 * totalContributionMax
                minPayoff2 = minPayoff2 * totalContributionMin

                fitness += fitness2
                maxPayoff += maxPayoff2
                minPayoff += minPayoff2

        return fitness, minPayoff, maxPayoff

    def imitationProbability(self, population, populationGraphIndices, node, neighbor):
        nodeFitness, nodeMinPayoff, nodeMaxPayoff = self.computeFitness(population, populationGraphIndices, node)
        neighborFitness, neighborMinPayoff, neighborMaxPayoff = self.computeFitness(population, populationGraphIndices, neighbor)
        if nodeFitness > neighborFitness:
            return -1
        M = max(abs(nodeMaxPayoff-neighborMinPayoff), abs(neighborMaxPayoff-nodeMinPayoff))

        return (neighborFitness-nodeFitness) / M

    def nextGen(self, population, populationGraphIndices):
        if self.evolutionModel == 0:  # pairwise comparison
            for node in populationGraphIndices.nodes():
                neighbor = np.random.choice(list(populationGraphIndices.neighbors(node)))
                if random.random() < self.imitationProbability(population, populationGraphIndices, node, neighbor):
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
        valuesPerGraph = np.zeros((self.graphNum, 2, self.genNum+1))
        valuesPerRun = np.zeros((self.runNum, 2, self.genNum+1))

        for graph in range(self.graphNum):
            # Create graph representing the population indices (will be used to represent the structure)
            populationGraphIndices = nx.random_regular_graph(self.graphConnectivity, self.populationSize)
            # Create and shuffle population (0: D, 1:C)
            ic(graph)

            for run in range(self.runNum):
                self.r = 1

                population = np.copy(populationOriginal)
                np.random.shuffle(population)
                for _ in range(self.transientGenNum):
                    self.nextGen(population, populationGraphIndices)
                    self.r += (self.graphConnectivity) / (self.genNum)

                valuesPerRun[run, 0, 0] = self.r / (self.graphConnectivity + 1)
                valuesPerRun[run, 1, 0] = np.count_nonzero(population) / self.populationSize

                for gen in range(self.genNum):
                    self.nextGen(population, populationGraphIndices)
                    self.r += (self.graphConnectivity) / (self.genNum)
                    valuesPerRun[run, 0, gen+1] = self.r/(self.graphConnectivity + 1)
                    valuesPerRun[run, 1, gen+1] = np.count_nonzero(population) / self.populationSize


            valuesPerGraph[graph] = np.mean(valuesPerRun, axis=0)
        simulationValuesForRegularGraph = np.mean(valuesPerGraph, axis=0)
        ic(simulationValuesForRegularGraph)
        return simulationValuesForRegularGraph

    def simulateWithWealth(self) -> np.array([[int], [int]]):
        """
        :return: [x_values: their fraction of the total wealth, y_values: number of individuals]
        """
        # TODO: implement the simulation for the regular graph, to track the evolution of the number of individual
        #  as a function of their fraction of the total wealth. You are encouraged to divide this function
        #  into smaller functions

        simulationValuesForRegularGraphWithWealth = np.array([[], []])
        return simulationValuesForRegularGraphWithWealth


    def extractIndividuals(self, population, indices):
        return [population[i] for i in indices]






