"""
Authors: Alnakeb Derar, Caiola Ludovica, Correa Catalina, Andres Antelo
Description: Reproducing the results and experimentation conducted in the scientific paper: "Social Diversity Promotes
The Emergence Of Cooperation In Public Goods Games".
"""
from regularGraph import RegularGraph
from scaleFreeGraph import ScaleFreeGraph
import matplotlib.pyplot as plt


def plotRegularAndScaleFreeGraphs(regularGraphPlotValues: [[int], [int]],
                                  scaleFreeGraphPlotValues: [[int], [int]]):
    plt.plot(regularGraphPlotValues[0], regularGraphPlotValues[1])
    plt.scatter(regularGraphPlotValues[0], regularGraphPlotValues[1], color='red')

    plt.plot(scaleFreeGraphPlotValues[0], scaleFreeGraphPlotValues[1])
    plt.scatter(scaleFreeGraphPlotValues[0], scaleFreeGraphPlotValues[1], color='blue')

    plt.xlabel('n')
    plt.ylabel('C fraction')
    plt.legend()
    plt.xlim(-0.1, 1.2)
    plt.ylim(-0.1, 1.2)
    plt.show()

def plotWealthDistributionForRegularAndScaleFreeGraphs(simulationValuesForRegularGraphWithWealthPerGame: [[int],[int]],
                                                       simulationValuesForScaleFreeGraphWithWealthPerGame: [[int],[int]],
                                                       simulationValuesForScaleFreeGraphWithWealthPerInd: [[int],[int]]):
    plt.figure(figsize=(8, 6))
    plt.bar(simulationValuesForRegularGraphWithWealthPerGame[0], simulationValuesForRegularGraphWithWealthPerGame[1], width=0.002, align='center')
    plt.bar(simulationValuesForScaleFreeGraphWithWealthPerGame[0], simulationValuesForScaleFreeGraphWithWealthPerGame[1], width=0.002, align='center')
    plt.scatter(simulationValuesForScaleFreeGraphWithWealthPerInd[0], simulationValuesForScaleFreeGraphWithWealthPerInd[1])
    plt.xlabel('Fraction of total wealth')
    plt.ylabel('Number of individuals')
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.axis([0.001, 0.2, 0.01, 1000])
    plt.show()


if __name__ == "__main__":
    populationSize = 5*10 ** 2
    transientGenNum = 5*10 ** 2
    genNum = 10 ** 1
    graphNum = 5
    runNum = 2
    initCooperatorsFraction = 0.5
    averageGraphConnectivity = 4
    contributionModel = 1  # (0: cost per game, 1: cost per individual)


    regularGraph = RegularGraph(populationSize, transientGenNum, genNum, graphNum, runNum, initCooperatorsFraction,
                                averageGraphConnectivity, contributionModel)

    scaleFreeGraph = ScaleFreeGraph(populationSize, transientGenNum, genNum, graphNum, runNum, initCooperatorsFraction,
                                    averageGraphConnectivity, contributionModel)

    simulationValuesForRegularGraph = regularGraph.simulate()
    simulationValuesForScaleFreeGraph = scaleFreeGraph.simulate()

    plotRegularAndScaleFreeGraphs(simulationValuesForRegularGraph, simulationValuesForScaleFreeGraph)

    """simulationValuesForRegularGraphWithWealthPerGame = regularGraph.simulateWithWealth(0)
    simulationValuesForScaleFreeGraphWithWealthPerGame = scaleFreeGraph.simulateWithWealth(0)
    simulationValuesForScaleFreeGraphWithWealthPerInd = scaleFreeGraph.simulateWithWealth(1)"""

    #plotWealthDistributionForRegularAndScaleFreeGraphs(simulationValuesForRegularGraphWithWealthPerGame,
    #                                                    simulationValuesForScaleFreeGraphWithWealthPerGame, simulationValuesForScaleFreeGraphWithWealthPerInd)