import random
from copy import copy
import numpy as np
from icecream import ic
import networkx as nx


class ScaleFreeGraph:
    """
    Class implementing a PGG simulation with a scale-free graph population.
    """
    def __init__(self, populationSize: int, transientGenNum: int, genNum: int, graphNum: int, runNum: int,
                 initCooperatorsFraction: float,
                 averageGraphConnectivity: int,
                 contributionValue: int,
                 contributionModel: int):
        """
        :param populationSize:  number of nodes in the graph
        :param transientGenNum:  transient number of generations before taking data
        :param genNum:  number of generations
        :param graphNum: number of instances of graphs of the same class
        :param runNum:  number of runs for each instance of a graph
        :param initCooperatorsFraction:  initial cooperators fraction in the population
        :param averageGraphConnectivity:  graph connectivity
        :param contributionValue:  contribution value
        :param contributionModel: (0: cost per game, 1: cost per individual)
        """
        self.populationSize = populationSize
        self.transientGenNum = transientGenNum
        self.genNum = genNum
        self.graphNum = graphNum
        self.runNum = runNum
        self.initCooperatorsFraction = initCooperatorsFraction
        self.averageGraphConnectivity = averageGraphConnectivity
        self.contributionValue = contributionValue
        self.contributionModel = contributionModel

        self.r = 1
        self.neighborsOf = {}
        self.wealthPerIndividual = None

    def computeFitness(self, population, node):
        """
        Computing fitness based on the accumulated payoff of an individual depending on the contribution model (fixed
         cost per individual or per game).
        :param population:
        :param node:
        :return: fitness, minimum possible payoff, maximum possible payoff
        """
        fitness = 0
        minPayoff = 0
        maxPayoff = 0

        if self.contributionModel == 0:  # fixed cost per game
            c = self.extractIndividuals(population, self.neighborsOf[node]).count(1) + population[node]
            k = len(self.neighborsOf[node])

            fitness += self.r * c / (k + 1) - population[node]
            minPayoff += self.r * (population[node]) / (k + 1) - population[node]
            maxPayoff += self.r * (len(self.neighborsOf[node]) + population[node]) / (k + 1) - population[node]

            for neighborIndex in self.neighborsOf[node]:
                c = self.extractIndividuals(population, self.neighborsOf[neighborIndex]).count(1) + population[neighborIndex]
                k = len(self.neighborsOf[neighborIndex])

                fitness += self.r * c / (k + 1) - population[node]
                minPayoff += self.r * (population[node]) / (k + 1) - population[node]
                maxPayoff += self.r * (len(self.neighborsOf[neighborIndex]) + 1 + population[node]) / (k + 1) - population[node]

        else:  # fixed cost per individual
            k_y = len(self.neighborsOf[node])
            fitness += self.r / (k_y+1)
            minPayoff += self.r / (k_y+1)
            maxPayoff += self.r / (k_y+1)

            totalContribution = (1/(k_y+1))*population[node]
            totalContributionMax = (1/(k_y+1))*population[node]
            totalContributionMin = (1/(k_y+1))*population[node]
            for neighborIndex in self.neighborsOf[node]:
                k_i = len(self.neighborsOf[neighborIndex])
                totalContribution += (1/(k_i+1))*population[neighborIndex]
                totalContributionMax += (1/(k_i+1))

            fitness = fitness*totalContribution - 1*population[node]
            maxPayoff = maxPayoff*totalContributionMax - 1*population[node]
            minPayoff = minPayoff*totalContributionMin - 1*population[node]

            for neighborIndex in self.neighborsOf[node]:
                neighborsIndices2 = self.neighborsOf[neighborIndex]
                k_x = len(neighborsIndices2)
                fitness2 = self.r / (k_x + 1)
                minPayoff2 = self.r / (k_x + 1)
                maxPayoff2 = self.r / (k_x + 1)

                totalContribution = (1 / (k_x + 1)) * population[neighborIndex]
                totalContributionMax = (1 / (k_x + 1))
                totalContributionMin = (1 / (k_x + 1)) * population[node]
                for neighborIndex2 in neighborsIndices2:
                    k_i = len(self.neighborsOf[neighborIndex2])
                    totalContribution += (1 / (k_i + 1)) * population[neighborIndex2]
                    totalContributionMax += (1 / (k_i + 1))

                fitness2 = fitness2 * totalContribution
                maxPayoff2 = maxPayoff2 * totalContributionMax
                minPayoff2 = minPayoff2 * totalContributionMin

                fitness += fitness2
                maxPayoff += maxPayoff2
                minPayoff += minPayoff2

        return fitness, minPayoff, maxPayoff

    def imitationProbability(self, population, node, neighbor):
        """
        Calculates the imitation probability for a node to his neighbor based on the normalized difference of fitness
        between both.
        :param population:
        :param node:
        :param neighbor:
        :return: normalized difference of fitness or -1 if node has better fitness.
        """
        nodeFitness, nodeMinPayoff, nodeMaxPayoff = self.computeFitness(population, node)
        neighborFitness, neighborMinPayoff, neighborMaxPayoff = self.computeFitness(population, neighbor)
        if nodeFitness > neighborFitness:
            return -1
        M = max(nodeMaxPayoff-neighborMinPayoff, neighborMaxPayoff-nodeMinPayoff)

        return (neighborFitness-nodeFitness) / M

    def nextGen(self, population):
        """
        Runs one iteration of the simulation to attain the next generation in the population
        :param population:
        :return: None
        """
        for node in self.neighborsOf:
            neighbor = random.choice(self.neighborsOf[node])
            if random.random() < self.imitationProbability(population, node, neighbor):
                population[node] = population[neighbor]

        return population

    def simulate(self) -> np.array([[int], [int]]):
        """
        Runs the simulation of the PGG given the parameters on the initiating of the class
        :return: [x_values: n=r/(z+1) renormalized PGG enhancement factor, y_values: fraction of cooperators]
        """
        cooperators = np.ones(int(self.populationSize * self.initCooperatorsFraction))
        defectors = np.zeros(int(self.populationSize * (1 - self.initCooperatorsFraction)))
        populationOriginal = np.hstack((cooperators, defectors))

        valuesPerGraphs = np.zeros((self.graphNum, 2, (self.averageGraphConnectivity + 1) * 5 + - 1))  # (number of graphs, [n, C fraction], number of fractions of r)
        valuesPerRuns = np.zeros((self.runNum, 2, (self.averageGraphConnectivity + 1) * 5 + - 1))  # (number of graphs, [n, C fraction], number of fractions or r)
        valuesPerGens = np.zeros(self.genNum)

        for r in range(1, (self.averageGraphConnectivity + 1) * 5 + 4):
            self.r = (r/5)
            ic(r)

            for graph in range(self.graphNum):
                # Create graph representing the population indices (will be used to represent the structure)
                populationGraphIndices = nx.barabasi_albert_graph(self.populationSize, self.averageGraphConnectivity)

                self.neighborsOf = {}
                for node in populationGraphIndices.nodes():  # for optimality reasons
                    self.neighborsOf[node] = list(populationGraphIndices.neighbors(node))
                for run in range(self.runNum):
                    #ic(run)
                    # Create and shuffle population (0: D, 1:C)
                    population = copy(populationOriginal)
                    random.shuffle(population)

                    for _ in range(self.transientGenNum):
                        self.nextGen(population)

                    for gen in range(self.genNum):
                        self.nextGen(population)
                        valuesPerGens[gen] = np.count_nonzero(population) / self.populationSize

                    valuesPerRuns[run, 0, r-5] = self.r/(self.averageGraphConnectivity + 1)
                    valuesPerRuns[run, 1, r-5] = np.mean(valuesPerGens)

                valuesPerGraphs[graph] = np.mean(valuesPerRuns, axis=0)

        simulationValuesForRegularGraph = np.mean(valuesPerGraphs, axis=0)
        ic(simulationValuesForRegularGraph)
        return simulationValuesForRegularGraph

    def computeFitnessWithWealth(self, population, node):
        """
        Computes the fitness of an individual
        :param population:
        :param node:
        :return: Fitness
        """
        fitness = 0
        neighborsIndices = self.neighborsOf[node]
        neighborsStrat = self.extractIndividuals(population, neighborsIndices)

        if self.contributionModel == 0:  # fixed cost per game
            c = neighborsStrat.count(1) + population[node]
            k = len(self.neighborsOf[node])

            fitness += self.r * c / (k + 1) - population[node]

            for neighborIndex in neighborsIndices:
                c = self.extractIndividuals(population, self.neighborsOf[neighborIndex]).count(1) + population[
                    neighborIndex]
                k = len(self.neighborsOf[neighborIndex])
                fitness += self.r * c / (k + 1) - population[node]

        else:  # same cost per individual
            k_y = len(self.neighborsOf[node])
            fitness += self.r / (k_y + 1)

            totalContribution = (1 / (k_y + 1)) * population[node]
            for neighborIndex in neighborsIndices:
                k_i = len(self.neighborsOf[neighborIndex])
                totalContribution += (1 / (k_i + 1)) * population[neighborIndex]

            fitness = fitness * totalContribution - 1 * population[node]

            for neighborIndex in neighborsIndices:
                neighborsIndices2 = self.neighborsOf[neighborIndex]
                k_x = len(neighborsIndices2)
                fitness2 = self.r / (k_x + 1)

                totalContribution = (1 / (k_x + 1)) * population[neighborIndex]
                for neighborIndex2 in neighborsIndices2:
                    k_i = len(self.neighborsOf[neighborIndex2])
                    totalContribution += (1 / (k_i + 1)) * population[neighborIndex2]

                fitness2 = fitness2 * totalContribution
                fitness += fitness2

        return fitness

    def nextGenWithWealth(self, population):
        """
        Runs one iteration of the simulation and sum the new fitness to the wealth of every individual
        :param population:
        :return: None
        """
        for node in self.neighborsOf:
            self.wealthPerIndividual[node] += self.computeFitnessWithWealth(population, node)

    def simulateWithWealth(self, contributionModel) -> np.array([[int], [int]]):
        """
        Runs the simulation with an entier population of cooperators and calculating the wealth of each at every run
        :return: [x_values: their fraction of the total wealth, y_values: number of individuals]
        """
        populationOriginal = np.ones(int(self.populationSize))
        self.wealthPerIndividual = np.zeros(int(self.populationSize))
        self.contributionModel = contributionModel

        valuesPerGraphs = np.zeros((self.graphNum, 2, self.populationSize))  # (number of graphs, [n, C fraction], number of fractions or r)
        valuesPerRuns = np.zeros((self.runNum, 2, self.populationSize))  # (number of graphs, [n, C fraction], number of fractions or r)
        self.r = 1.25

        for graph in range(self.graphNum):
            # Create graph representing the population indices (will be used to represent the structure)
            populationGraphIndices = nx.barabasi_albert_graph(self.populationSize, self.averageGraphConnectivity)
            ic(graph)
            self.neighborsOf = {}
            for node in populationGraphIndices.nodes():  # for optimality reasons
                self.neighborsOf[node] = list(populationGraphIndices.neighbors(node))

            for run in range(self.runNum):
                self.wealthPerIndividual = np.zeros(int(self.populationSize))
                # Create and shuffle population (0: D, 1:C)
                population = copy(populationOriginal)
                random.shuffle(population)
                #ic(run)
                for _ in range(self.transientGenNum):
                    self.nextGenWithWealth(population)

                total = np.sum(self.wealthPerIndividual)
                self.wealthPerIndividual = self.wealthPerIndividual / total
                fractionOfTotalWealth, numberOfIndividuals = np.unique(np.round(self.wealthPerIndividual, 2), return_counts=True)
                ic(fractionOfTotalWealth)
                ic(numberOfIndividuals)

                valuesPerRuns[run, 0, 0:len(fractionOfTotalWealth)] = fractionOfTotalWealth
                valuesPerRuns[run, 1, 0:len(numberOfIndividuals)] = numberOfIndividuals

            valuesPerGraphs[graph] = np.mean(valuesPerRuns, axis=0)

        simulationValuesForRegularGraph = np.mean(valuesPerGraphs, axis=0)
        ic(simulationValuesForRegularGraph)
        return simulationValuesForRegularGraph

    def extractIndividuals(self, population, indices):
        return [population[i] for i in indices]














