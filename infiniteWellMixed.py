import egttools as egt
from egttools.plotting.simplified import plot_replicator_dynamics_in_simplex
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic


class InfiniteWellMixed:
    """
    Class implementing an infinite well mixed population using egttools library. With the option of plotting it.
    """
    def __init__(self, cooperator: egt.behaviors.NormalForm.TwoActions.Cooperator(),
                 defector: egt.behaviors.NormalForm.TwoActions.Defector(), payoffMatrix: [[int], [int]], nbRounds: int):
        """
        :param cooperator: cooperator strategy object from egttools
        :param defector: defector strategy object from egttools
        :param payoffMatrix: payoffMatrix of the game ([[1,1], [1,1]])?
        """
        self.cooperator = cooperator
        self.defector = defector
        self.payoffMatrix = payoffMatrix
        self.nbRounds = nbRounds

    def simulate(self) -> np.array([[int],[int]]):
        """
        runs a simulation given 2 strategies (Cooperator and Defector) in an infinite well mixed population and a
        payoff matrix.
        :return: [x_values: n=r/(z+1) renormalized PGG enhancement factor, y_values: fraction of cooperators]
        """
        self.payoffMatrix = np.array(self.payoffMatrix)
        strategies = [self.cooperator,
                      self.defector]
        #strategy_labels = [strategy.type().replace("NFGStrategies::", '') for strategy in strategies]
        valuesPerFractionR = np.zeros(1)

        for r in range(1):
            game = egt.games.NormalFormGame(self.nbRounds, self.payoffMatrix, strategies)
            fig, ax = plt.subplots(figsize=(10, 8))

            simplex, gradients, roots, roots_xy, stability = plot_replicator_dynamics_in_simplex(game.expected_payoffs(), ax=ax)

        simulationValuesForInfiniteWellMixed = np.mean(valuesPerFractionR, axis=0)
        return simulationValuesForInfiniteWellMixed

    def plot(self, game: egt.games.NormalFormGame):
        """
        Plot in a replicator dynamics simplex graph the fraction of cooperators as a function of the
        gradient of selection.
        :param game: a normal forme game state
        :return None
        """
        # TODO: use plot_replicator_dynamics_in_simplex
        pass
