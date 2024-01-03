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


def plotRegularAndScaleFreeGraphs(regularGraphPlotValues: [[int], [int]]):
    # TODO: implement the graph plotting for both classes of graph the fraction of cooperators as a function of n=r(z+1)
    plt.plot(regularGraphPlotValues[0], regularGraphPlotValues[1])
    plt.xlabel('n')
    plt.ylabel('C fraction')
    plt.title('Plotting data')
    plt.legend()
    plt.xlim(0.2,1)
    plt.ylim(0,1.2)
    plt.show()


def plotWealthDistributionForRegularAndScaleFreeGraphs(regularGraphWealthPlotValues: [[int], [int]],
                                                       scaleFreeGraphWealthPlotValues: [[int], [int]]):
    # TODO: implement the graph for plotting the number of individuals as a function of their fraction
    #  of the total wealth
    pass


if __name__ == "__main__":
    payoffMatrix = [[], []]  # to be determined ???
    populationSize = 10**2
    transientGenNum = 3*10**2
    genNum = 2*10**1
    graphNum = 5
    runNum = 10
    initCooperatorsFraction = 0.5
    averageGraphConnectivity = 4
    contributionValue = 20
    contributionModel = 1  # (0: cost per game, 1: cost per individual)
    evolutionModel = 0  # (0: pairwise comparison, 1: death-birth , 2: birth-death)
    updateStrategy = 0  # (0: synchronous, 1: asynchronous)
    mutations = False  # true if mutations are allowed

    #infiniteWellMixed = InfiniteWellMixed(egt.behaviors.NormalForm.TwoActions.Cooperator(),
    #                                      egt.behaviors.NormalForm.TwoActions.Defector(), payoffMatrix)
    #infiniteWellMixed.plot(infiniteWellMixed.simulate())

    regularGraph = RegularGraph(populationSize, transientGenNum, genNum, graphNum, runNum, initCooperatorsFraction,
                               averageGraphConnectivity, contributionValue, contributionModel, evolutionModel, updateStrategy, mutations)
#
   # scaleFreeGraph = ScaleFreeGraph(populationSize, transientGenNum, genNum, graphNum, runNum, initCooperatorsFraction,
   #                        averageGraphConnectivity, contributionValue, contributionModel, evolutionModel, updateStrategy, mutations)
#
    simulationValuesForRegularGraph = regularGraph.simulate()
    #simulationValuesForScaleFreeGraph = scaleFreeGraph.simulate()
    plotRegularAndScaleFreeGraphs(simulationValuesForRegularGraph)

    #simulationValuesForRegularGraphWithWealth = regularGraph.simulateWithWealth()
    #simulationValuesForScaleFreeGraphWithWealth = regularGraph.simulateWithWealth()
    #plotWealthDistributionForRegularAndScaleFreeGraphs(simulationValuesForRegularGraphWithWealth,
    #                                                    simulationValuesForScaleFreeGraphWithWealth)