__author__ = 'bryantb'
from abc import ABCMeta, abstractmethod

class Strategy(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def generate_signals(selfself):
        raise NotImplementedError("Should implement generate_signals()!")


class Portfolio(object):
    __metaclass__ = ABCMeta
    @abstractmethod
    def generate_positions(selfself):
        raise NotImplementedError("Should implement generate_positions()!")

    @abstractmethod
    def backtest_portfolio(self):
        raise NotImplemented("Should implement backtest_portfolio()!")

