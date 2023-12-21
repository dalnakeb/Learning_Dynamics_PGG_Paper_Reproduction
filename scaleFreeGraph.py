import egttools as egt
import numpy as np
from icecream import ic
import networkx as nx


class ScaleFreeGraph:
    """
    Class implementing a scale free graph population.
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
        # TODO: initialize more attributes if needed

    def simulate(self) -> np.array([[int], [int]]):
        """
        :return: [x_values: n=r/(z+1) renormalized PGG enhancement factor, y_values: fraction of cooperators]
        """
        # TODO: implement the simulation for the scale free graph, to track the evolution of the fraction of cooperators
        #  as a function of the renormalized PGG enhancement factor. You are encouraged to divide this function
        #  into smaller functions
        pass

    def simulateWithWealth(self) -> np.array([int], [int]):
        """
        :return: [x_values: their fraction of the total wealth, y_values: number of individuals]
        """
        # TODO: implement the simulation for the scale free graph, to track the evolution of the number of individual
        #  as a function of their fraction of the total wealth. You are encouraged to divide this function
        #  into smaller functions
        pass