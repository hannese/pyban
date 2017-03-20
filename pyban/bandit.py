import numpy as np


class Bandit(object):
    def __init__(self):
        self.mean = 0
        self.std = 1


class BernoulliBandit(Bandit):
    def __init__(self, p):
        super(BernoulliBandit, self).__init__()
        self.p = p

    def sample(self):
        return np.random.binomial(1, self.p)

    def update(self):
        self.mean = self.p
        self.std = self.p * (1 - self.p)


class ContextualBernoulliBandit(Bandit):
    def __init__(self, mf):
        super(ContextualBernoulliBandit, self).__init__()
        self.mf = mf

    def sample(self, X):
        return np.random.binomial(1, self.mf(X))

    def update(self):
        pass



