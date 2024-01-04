from regularGraph import RegularGraph
from scaleFreeGraph import ScaleFreeGraph
from icecream import ic
import matplotlib.pyplot as plt

if __name__ == "__main__":
    populationSize = 10**2
    transientGenNum = 10**3
    genNum = 10**1
    graphNum = 1
    runNum = 1
    initCooperatorsFraction = 0.5
    averageGraphConnectivity = 4
    contributionValue = 0
    contributionModel = 1  # (0: cost per game, 1: cost per individual)

    regularGraph = RegularGraph(populationSize, transientGenNum, genNum, graphNum, runNum, initCooperatorsFraction,
                                averageGraphConnectivity, contributionValue, contributionModel)
    scaleFreeGraph = ScaleFreeGraph(populationSize, transientGenNum, genNum, graphNum, runNum, initCooperatorsFraction,
                                averageGraphConnectivity, contributionValue, contributionModel)
    values = regularGraph.simulate()
    #values = scaleFreeGraph.simulate()
    #values = regularGraph.simulateWithWealth()

    """plt.figure(figsize=(8, 6))
    plt.bar(values[0], values[1], width=0.002, align='center')  # Adjust width as needed
    plt.xlabel('Fraction of total wealth')
    plt.ylabel('Number of individuals')
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.axis([0.001, 0.2, 0.01, 1000])
    plt.show()"""

    plt.plot(values[0], values[1])
    plt.scatter(values[0], values[1], color='red')
    plt.xlabel('n')
    plt.ylabel('C fraction')
    plt.legend()
    plt.xlim(0.2, 1.2)
    plt.ylim(-0.1, 1.2)
    plt.show()
