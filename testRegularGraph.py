from regularGraph import RegularGraph
from icecream import ic
import matplotlib.pyplot as plt


if __name__ == "__main__":
    populationSize = 30
    transientGenNum = 1
    genNum = 10**2
    graphNum = 4
    runNum = 10
    initCooperatorsFraction = 0.5
    averageGraphConnectivity = 4
    contributionValue = 1
    contributionModel = 1  # (0: cost per game, 1: cost per individual)

    regularGraph = RegularGraph(populationSize, transientGenNum, genNum, graphNum, runNum, initCooperatorsFraction,
                                averageGraphConnectivity, contributionValue, contributionModel)

    values = regularGraph.simulate()

    plt.plot(values[0], values[1])
    plt.xlabel('n')
    plt.ylabel('C fraction')
    plt.title('Plotting Data')
    plt.legend()
    #plt.xlim(0, 1.1)
    #plt.ylim(0, 1.1)
    plt.show()

