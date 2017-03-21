import numpy as np
from scipy.special import gamma
import math
import GPy

class Policy(object):
    def __init__(self):
        self.rewards = []

# Plays the bandit with the highest mean
class OracleBest(Policy):
    def __init__(self):
        super(OracleBest, self).__init__()
        self.name = "ContextualOracleBest"

    def play(self, bandits, x):
        return np.argmax([b.mf(x) for b in bandits])

    def observe(self, reward, i):
        self.rewards.append(reward)

# Plays the bandit with the lowest mean
class OracleWorst(Policy):
    def __init__(self):
        super(OracleWorst, self).__init__()
        self.name = "ContextualOracleWorst"

    def play(self, bandits, x):
        return np.argmin([b.mf(x) for b in bandits])

    def observe(self, reward, i):
        self.rewards.append(reward)


class Random(Policy):
    def __init__(self):
        super(Random, self).__init__()
        self.name = "Random"

    def play(self, bandits, x):
        return np.random.randint(0, len(bandits))

    def observe(self, reward, i):
        self.rewards.append(reward)


class GP_UCB(Policy):
    def __init__(self, n, kernel, likelihood, delta):
        super(GP_UCB, self).__init__()
        self.name = "ContextualGP-UCB"
        self.gps = [None] * n
        self.n = n
        self.d = None
        self.t = 1
        self.kernel = kernel
        self.likelihood = likelihood
        self.delta = delta

        self.Xk = np.array([[[]]])
        self.Yk = np.array([[[]]])

    def __beta__(self):
        return 2. * math.log(math.fabs(self.Xk.shape[1]) * self.t ** 2 * math.pi ** 2 / (6 * self.delta))

    def play(self, bandits, x):
        if self.t < self.n * 10:
            idx = (self.n * 10) % self.t
            self.d = x.shape[0]
            return idx

        vals = []
        for gp in self.gps:
            mean, var = gp.predict_noiseless(x)
            vals.append(mean + math.sqrt(self.__beta__() * var))

        return np.argmax(vals)

    def observe(self, reward, i, x):
        self.Xk[i].append(x)
        self.Yk[i].append(reward)
        self.gps[i].set_XY(self.Xk[i], self.Yk[i])


class GP_TS(Policy):
    def __init__(self, n, kernel, likelihood, delta):
        super(GP_TS, self).__init__()
        self.name = "ContextualGP-Thompson"
        self.gps = [None] * n
        self.n = n
        self.d = None
        self.t = 1
        self.kernel = kernel
        self.likelihood = likelihood
        self.delta = delta

        self.Xk = np.array([[[]]])
        self.Yk = np.array([[[]]])

    def play(self, x):
        if self.t < self.n * 10:
            idx = (self.n * 10) % self.t
            self.d = x.shape[0]
            return idx

        vals = []
        for gp in self.gps:
            mean, var = gp.predict_noiseless(x)
            vals.append(np.random.normal(mean, var))

        return np.argmax(vals)

    def observe(self, reward, i, x):
        self.Xk[i].append(x)
        self.Yk[i].append(reward)
        self.gps[i].set_XY(self.Xk[i], self.Yk[i])

class GP_TSTree(Policy):
    def __init__(self, n, d, kernel, likelihood, delta):
        super(GP_TSTree, self).__init__()
        self.name = "ContextualGP-ThompsonTree"
    def play(self, bandits):
        pass
    def observe(self, reward, i, X):
        pass

