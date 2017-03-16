import numpy as np


class Policy(object):
    def __init__(self):
        self.rewards = []


class OracleBest(Policy):
    def __init__(self):
        super(OracleBest, self).__init__()
        self.name = "OracleBest"

    def play(self, bandits):
        return np.argmax([b.mean for b in bandits])

    def observe(self, reward, i):
        self.rewards.append(reward)


class OracleWorst(Policy):
    def __init__(self):
        super(OracleWorst, self).__init__()
        self.name = "OracleWorst"

    def play(self, bandits):
        return np.argmin([b.mean for b in bandits])

    def observe(self, reward, i):
        self.rewards.append(reward)


class Random(Policy):
    def __init__(self):
        super(Random, self).__init__()
        self.name = "Random"

    def play(self, bandits):
        return np.random.randint(0, len(bandits))

    def observe(self, reward, i):
        self.rewards.append(reward)


class BetaThompson(Policy):
    def __init__(self, n):
        super(BetaThompson, self).__init__()
        self.beta_paras = np.ones((n, 2)) * 0.5
        self.name = "BetaThompson"

    def play(self, bandits):
        return np.argmax([np.random.beta(par[0], par[1]) for par in self.beta_paras])

    def observe(self, reward, i):
        if reward == 1:
            self.beta_paras[i][0] += 1
        else:
            self.beta_paras[i][1] += 1
        self.rewards.append(reward)


