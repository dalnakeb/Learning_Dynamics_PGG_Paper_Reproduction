import egttools as egt


class InfiniteWellMixed:
    """
    Class implementing an infinite well mixed population using egttools library. With the option of plotting it.
    """
    def __init__(self, cooperator: egt.behaviors.NormalForm.TwoActions.Cooperator,
                 defector: egt.behaviors.NormalForm.TwoActions.Defector, payoffMatrix: [[int]]):
        """
        :param cooperator: cooperator strategy object from egttools
        :param defector: defector strategy object from egttools
        :param payoffMatrix: payoffMatrix of the game ([[1,1], [1,1]])?
        """
        self.cooperator = cooperator
        self.defector = defector
        self.payoffMatrix = payoffMatrix

        # TODO: initialize more attributes if needed

    def simulate(self, nbRounds: int) -> egt.games.NormalFormGame:
        """
        runs a simulation given 2 strategies (Cooperator and Defector) in an infinite well mixed population and a
        payoff matrix.
        :param nbRounds: number of rounds to run in the simulation
        :return: a normal form game state after the simulation
        """
        # TODO: run the simulation with nbRounds rounds. You are encouraged to divide this
        #  function into smaller function
        pass

    def plot(self, game: egt.games.NormalFormGame):
        """
        Plot in a replicator dynamics simplex graph the fraction of cooperators as a function of the
        gradient of selection.
        :param game: a normal forme game state
        :return None
        """
        # TODO: use plot_replicator_dynamics_in_simplex
        pass
