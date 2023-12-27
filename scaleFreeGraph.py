import egttools as egt
import numpy as np
from icecream import ic
import networkx as nx
import random


class ScaleFreeGraph:
    """
    Class implementing a scale free graph population.
    """

    def __init__(self, populationSize: int, transientGenNum: int, genNum: int, graphNum: int, runNum: int,
                 initCooperatorsFraction: float,
                 averageGraphConnectivity: int,
                 contributionValue: int,
                 contributionModel: int,
                 payoffMatrix: np.array):
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
        self.payoffMatrix = payoffMatrix
        # TODO: initialize more attributes if needed

        # Creation of scale free network
        self.G = nx.barabasi_albert_graph(self.populationSize, self.averageGraphConnectivity)

    def payoffSubstraction(self, currentPopulation, node, neighbour):
        """
        :param currentPopulation:
        :param node:
        :param neighbour:
        :return: return
        """

        payoff0 = 0
        payoff1 = 0
        for n in list(self.G.neighbors(node)):
            payoff0 += self.payoffMatrix[currentPopulation[node], currentPopulation[n]]
        for m in list(self.G.neighbors(neighbour)):
            payoff1 += self.payoffMatrix[currentPopulation[neighbour], currentPopulation[m]]
        return ((payoff1 / len(list(self.G.neighbors(neighbour)))) - (payoff0 / len(list(self.G.neighbors(node)))))

    def simulate(self) -> np.array([[int], [int]]):
        """
        :return: [x_values: n=r/(z+1) renormalized PGG enhancement factor, y_values: fraction of cooperators]
        """
        # TODO: implement the simulation for the scale free graph, to track the evolution of the fraction of cooperators
        #  as a function of the renormalized PGG enhancement factor. You are encouraged to divide this function
        #  into smaller functions

        # Definition of strategies and dictionary
        strategies = ['Cooperator', 'Defector']
        stratNum = len(strategies)
        strategy_dict = {i: strategies[i] for i in range(stratNum)}

        # Initialiaze population with strategies
        population = np.repeat(list(strategy_dict.keys()), self.populationSize)

        random.shuffle(population)

        # initialize an array to store the results
        collect_results = np.zeros((self.runNum + 1, len(strategies), self.genNum + 1))
        collect_results[0, :, 0] = np.bincount(population, minlength=stratNum) / self.populationSize

        # run the game for 100 independent runs
        for run in range(self.runNum):
            currentPopulation = population.copy()
            collect_results[run + 1, :, 0] = np.bincount(currentPopulation, minlength=1) / self.populationSize
            # TODO: check the minlength value for counting just the number of cooperators


            # run the game for 100 generations
            for generation in range(self.genNum):
                # update the strategies
                for node in self.G.nodes():

                    # Choose a random neighbour
                    neighbour = random.choice(list(self.G.neighbors(node)))

                    # update the strategy with the Fermi probability
                    if np.random.rand() < self.payoffSubstraction(currentPopulation, node, neighbour):
                        # TODO: normalize the result of the function payoff substraction
                        currentPopulation[node] = currentPopulation[neighbour]

                collect_results[run + 1, :, generation + 1] = np.bincount(currentPopulation,
                                                                          minlength=1) / self.populationSize

    def simulateWithWealth(self) -> np.array([int], [int]):
        """
        :return: [x_values: their fraction of the total wealth, y_values: number of individuals]
        """
        # TODO: implement the simulation for the scale free graph, to track the evolution of the number of individual
        #  as a function of their fraction of the total wealth. You are encouraged to divide this function
        #  into smaller functions
        pass

    def computeEta(self, r):
        return r / (self.averageGraphConnectivity + 1)