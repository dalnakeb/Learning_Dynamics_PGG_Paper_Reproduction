"""
Authors: Alnakeb Derar, Caiola Ludovica, Correa Catalina, Andres Antelo
Description: Reproducing the results and experimentation conducted in the scientific paper: "Social Diversity Promotes
The Emergence Of Cooperation In Public Goods Games".
"""
import numpy as np
from icecream import ic
import egttools as egt
from infiniteWellMixed import InfiniteWellMixed
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
    plt.xlim(0.2, 1.2)
    plt.ylim(0.2, 1.2)
    plt.show()


def plotWealthDistributionForRegularAndScaleFreeGraphs(regularGraphWealthPlotValues: [[int], [int]],
                                                       scaleFreeGraphWealthPlotValues: [[int], [int]]):
    plt.plot(regularGraphWealthPlotValues[0], regularGraphWealthPlotValues[1])
    plt.scatter(regularGraphWealthPlotValues[0], regularGraphWealthPlotValues[1], color='red')

    plt.plot(scaleFreeGraphWealthPlotValues[0], scaleFreeGraphWealthPlotValues[1])
    plt.scatter(scaleFreeGraphWealthPlotValues[0], scaleFreeGraphWealthPlotValues[1], color='blue')

    plt.xlabel('Number of individuals')
    plt.ylabel('Fraction of total wealth')
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    #plt.axis([0.0001, 0.2, 0.01, 1000])
    plt.show()


if __name__ == "__main__":
    payoffMatrix = [[], []]  # to be determined ???
    populationSize = 10 ** 3
    transientGenNum = 10 ** 5
    genNum = 2000
    graphNum = 10
    runNum = 100
    initCooperatorsFraction = 0.5
    averageGraphConnectivity = 4
    contributionValue = 1
    contributionModel = 0  # (0: cost per game, 1: cost per individual)
    evolutionModel = 0  # (0: pairwise comparison, 1: death-birth , 2: birth-death)
    updateStrategy = 0  # (0: synchronous, 1: asynchronous)
    mutations = False  # true if mutations are allowed

    #infiniteWellMixed = InfiniteWellMixed(egt.behaviors.NormalForm.TwoActions.Cooperator(),
    #                                      egt.behaviors.NormalForm.TwoActions.Defector(), payoffMatrix)
    #simulationValuesForInfiniteWellMixed = infiniteWellMixed.simulate()
    #infiniteWellMixed.plot(simulationValuesForInfiniteWellMixed)

    #regularGraph = RegularGraph(populationSize, transientGenNum, genNum, graphNum, runNum, initCooperatorsFraction,
    #                            averageGraphConnectivity, contributionValue, contributionModel, evolutionModel, updateStrategy, mutations)

    #scaleFreeGraph = ScaleFreeGraph(populationSize, transientGenNum, genNum, graphNum, runNum, initCooperatorsFraction,
    #                                averageGraphConnectivity, contributionValue, contributionModel)

    #simulationValuesForRegularGraph = regularGraph.simulate()
    #simulationValuesForScaleFreeGraph = scaleFreeGraph.simulate()
    #plotRegularAndScaleFreeGraphs(simulationValuesForRegularGraph, simulationValuesForScaleFreeGraph)

    #simulationValuesForRegularGraphWithWealth = regularGraph.simulateWithWealth()
    #simulationValuesForScaleFreeGraphWithWealth = regularGraph.simulateWithWealth()
    #plotWealthDistributionForRegularAndScaleFreeGraphs(simulationValuesForRegularGraphWithWealth,
    #                                                    simulationValuesForScaleFreeGraphWithWealth)