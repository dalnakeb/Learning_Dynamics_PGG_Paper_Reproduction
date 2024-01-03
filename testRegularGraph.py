import time

from regularGraph import RegularGraph
from icecream import ic
import matplotlib.pyplot as plt


if __name__ == "__main__":
    populationSize = 10**2
    transientGenNum = 7*10**3
    genNum = 10**1
    graphNum = 1
    runNum = 1
    initCooperatorsFraction = 0.5
    averageGraphConnectivity = 4
    contributionValue = 1
    contributionModel = 0  # (0: cost per game, 1: cost per individual)
    evolutionModel = 0  # (0: pairwise comparison, 1: death-birth , 2: birth-death)
    updateStrategy = 0  # (0: synchronous, 1: asynchronous)
    mutations = False  # true if mutations are allowed

    regularGraph = RegularGraph(populationSize, transientGenNum, genNum, graphNum, runNum, initCooperatorsFraction,
                                averageGraphConnectivity, contributionValue, contributionModel, evolutionModel, updateStrategy, mutations)

    values = regularGraph.simulate()

    plt.plot(values[0], values[1])
    plt.scatter(values[0], values[1], color='red')
    plt.xlabel('n')
    plt.ylabel('C fraction')
    plt.legend()
    plt.xlim(0.2, 1.2)
    plt.ylim(0.2, 1.2)
    plt.show()

