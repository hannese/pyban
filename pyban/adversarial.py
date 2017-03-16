import numpy as np
from scipy.special import gamma


class Policy(object):
    def __init__(self):
        self.rewards = []


class OracleBest(Policy):
    def __init__(self):
        super(OracleBest, self).__init__()
        self.name = "AdversarialOracleBest"

    def play(self, bandits):
        return np.argmax([b.mean for b in bandits])

    def observe(self, reward, i):
        self.rewards.append(reward)


class OracleWorst(Policy):
    def __init__(self):
        super(OracleWorst, self).__init__()
        self.name = "AdversarialOracleWorst"

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
        assert(n==2) #only supports 2 bandits for now ...
        super(BetaThompson, self).__init__()
        self.beta_paras = np.ones((n, 2))
        a, b, c, d = self.beta_paras.ravel()
        self.name = "AdversarialBetaThompson"
        self.pxy_t = gamma(a+b) * gamma(a+c) / (gamma(a+b+c) * gamma(a))

    def h(self):
        a, b, c, d = self.beta_paras.astype(int).ravel()

        return gamma(a+c) * gamma(b+d) * gamma(a+b) * gamma(c+d) / \
               (gamma(a) + gamma(b) + gamma(c) + gamma(d) + gamma(a+b+c+d))

    def play(self, bandits):
        return np.argmax([np.random.beta(par[0]//1, par[1]//1) for par in self.beta_paras])

    def observe(self, reward, i):
        a, b, c, d = self.beta_paras.ravel()
        if i == 0:
            if reward == 1: #a++
                self.beta_paras[i][0] += 1# * self.pxy_t
                test = self.h()/float(a)
                self.pxy_t += self.h()/float(a)
            else:           #b++
                self.beta_paras[i][1] += 1# * self.pxy_t
                self.pxy_t -= self.h()/float(b)
        else:
            if reward == 1: #c++
                self.beta_paras[i][0] += 1# * (1-self.pxy_t)
                self.pxy_t -= self.h()/float(c)
            else:           #d++
                self.beta_paras[i][1] += 1# * (1-self.pxy_t)
                self.pxy_t += self.h()/float(d)
        self.rewards.append(reward)


class Exp3(Policy):
    def __init__(self, n, gamma):
        super(Exp3, self).__init__()
        self.weights = np.ones((n, 1))
        self.gamma = gamma
        self.name = "AdversarialExp3"
        self.p_i = []
        self.n = n

    def play(self, bandits):
        self.p_i = [((1.0 - self.gamma) * w_j / np.sum(self.weights) + self.gamma / float(self.n))[0] for w_j in self.weights]
        return np.random.choice(range(self.n), 1, p=self.p_i)[0]

    def observe(self, reward, i):
        reward_ = reward / self.p_i[i]
        self.rewards.append(reward)
        test = self.gamma * reward_ / float(self.n)
        self.weights[i] *= np.exp(self.gamma * reward_ / float(self.n))

