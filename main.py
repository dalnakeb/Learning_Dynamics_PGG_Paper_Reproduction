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


def plotRegularAndScaleFreeGraphs(regularGraphPlotValues: [[int], [int]],
                                  scaleFreeGraphPlotValues: [[int], [int]]):
    # TODO: implement the graph plotting for both classes of graph the fraction of cooperators as a function of n=r(z+1)
    pass


def plotWealthDistributionForRegularAndScaleFreeGraphs(regularGraphWealthPlotValues: [[int], [int]],
                                                       scaleFreeGraphWealthPlotValues: [[int], [int]]):
    # TODO: implement the graph for plotting the number of individuals as a function of their fraction
    #  of the total wealth
    pass


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

    #infiniteWellMixed = InfiniteWellMixed(egt.behaviors.NormalForm.TwoActions.Cooperator(),
    #                                      egt.behaviors.NormalForm.TwoActions.Defector(), payoffMatrix)
    #simulationValuesForInfiniteWellMixed = infiniteWellMixed.simulate()
    #infiniteWellMixed.plot(simulationValuesForInfiniteWellMixed)

    #regularGraph = RegularGraph(populationSize, transientGenNum, genNum, graphNum, runNum, initCooperatorsFraction,
    #                            averageGraphConnectivity, contributionValue, contributionModel)

    #scaleFreeGraph = ScaleFreeGraph(populationSize, transientGenNum, genNum, graphNum, runNum, initCooperatorsFraction,
    #                                averageGraphConnectivity, contributionValue, contributionModel)

    #simulationValuesForRegularGraph = regularGraph.simulate()
    #simulationValuesForScaleFreeGraph = scaleFreeGraph.simulate()
    #plotRegularAndScaleFreeGraphs(simulationValuesForRegularGraph, simulationValuesForScaleFreeGraph)

    #simulationValuesForRegularGraphWithWealth = regularGraph.simulateWithWealth()
    #simulationValuesForScaleFreeGraphWithWealth = regularGraph.simulateWithWealth()
    #plotWealthDistributionForRegularAndScaleFreeGraphs(simulationValuesForRegularGraphWithWealth,
    #                                                    simulationValuesForScaleFreeGraphWithWealth)