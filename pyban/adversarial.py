import numpy as np
from scipy.special import gamma
import math


class Policy(object):
    def __init__(self):
        self.rewards = []

# Plays the bandit with the highest mean
class OracleBest(Policy):
    def __init__(self):
        super(OracleBest, self).__init__()
        self.name = "AdversarialOracleBest"

    def play(self, bandits):
        return np.argmax([b.mean for b in bandits])

    def observe(self, reward, i):
        self.rewards.append(reward)

# Plays the bandit with the lowest mean
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
    def __init__(self, n, gamma_const):
        assert(n==2) #only supports 2 bandits for now ...
        super(BetaThompson, self).__init__()
        self.gamma_const = gamma_const
        self.n = n
        self.beta_paras = np.ones((n, 2))
        a, b, c, d = self.beta_paras.ravel()
        self.name = "AdversarialBetaThompson"
        self.pxy_t = gamma(a+b) * gamma(a+c) / (gamma(a+b+c) * gamma(a))
        self.h_t = 1.0 / 6

    def play(self, bandits):
        return np.argmax([np.random.beta(par[0]//1, par[1]//1) for par in self.beta_paras])

    def observe(self, reward, i):
        a, b, c, d = self.beta_paras.ravel()
        a_, b_, c_, d_ = self.beta_paras.astype(int).ravel()
        self.rewards.append(reward)
        if i == 0:
            p = self.gamma_const / float(self.n) + (1-self.gamma_const) * self.pxy_t
            if reward == 1: #a++
                new_a = self.beta_paras[i][0] + math.sqrt(self.gamma_const * reward / (float(self.n) * p))
                #new_a = self.beta_paras[i][0] + reward / self.pxy_t * self.gamma_const / float(self.n)
                self.beta_paras[i][0] = new_a
                if new_a - a > 1.0:
                    self.h_t *= (a_ * c_) * (a_ * b_) / ((a_) * a_ * (b_ + c_ + d_))
                    self.pxy_t += self.h_t / float(a_)
            else:           #b++
                new_b = self.beta_paras[i][1] + math.sqrt(self.gamma_const * (1-reward) / (float(self.n) * p))
                #new_b = self.beta_paras[i][1] + (1-reward) / self.pxy_t * self.gamma_const / float(self.n)
                self.beta_paras[i][1] = new_b
                if new_b - b > 1.0:
                    self.h_t *= (b_ * d_) * (a_ * b_) / ((b_) * b_ * (a_ + c_ + d_))
                    self.pxy_t -= self.h_t / float(b_)
        else:
            p = self.gamma_const / float(self.n) + (1-self.gamma_const) * (1-self.pxy_t)
            if reward == 1: #c++
                new_c = self.beta_paras[i][0] + math.sqrt(self.gamma_const * reward / (float(self.n) * p))
                #new_c = self.beta_paras[i][0] + reward / self.pxy_t * self.gamma_const / float(self.n)
                self.beta_paras[i][0] = new_c
                if new_c - c > 1.0:
                    self.h_t *= (a_ * c_) * (c_ * d_) / ((c_) * c_ * (a_ + b_ + d_))
                    self.pxy_t -= self.h_t / float(c_)
            else:           #d++
                new_d = self.beta_paras[i][1] + math.sqrt(self.gamma_const * (1-reward) / (float(self.n) * p))
                #new_d = self.beta_paras[i][1] + (1-reward) / self.pxy_t * self.gamma_const / float(self.n)
                self.beta_paras[i][1] = new_d
                if new_d - d > 1.0:
                    self.h_t *= (b_ * d_) * (c_ * d_) / ((d_) * d_ * (a_ + b_ + c_))
                    self.pxy_t -= self.h_t / float(d_)

# The non-stochastic multi-armed bandit problem
# by P. Auer et al.
# https://cseweb.ucsd.edu/~yfreund/papers/bandits.pdf
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
        self.weights[i] *= np.exp(self.gamma * reward_ / float(self.n))

# Explore no more: Improved high-probability regret
#   bounds for non-stochastic bandits
# by Gergely Neu
# https://arxiv.org/pdf/1506.03271.pdf
# Idea: a time dependent gamma
class Exp3_IX(Policy):
    def __init__(self, n):
        super(Exp3_IX, self).__init__()
        self.weights = np.ones((n, 1))
        self.t = 1
        self.name = "AdversarialExp3_IX"
        self.p_i = []
        self.n = n

    def __gamma__(self):
        return math.sqrt(math.log(self.n) / (self.n * self.t))

    def play(self, bandits):
        self.p_i = [(w_j / np.sum(self.weights))[0] for w_j in self.weights]
        return np.random.choice(range(self.n), 1, p=self.p_i)[0]

    def observe(self, reward, i):
        loss = (1 - reward) / (self.p_i[i] + self.__gamma__())
        self.rewards.append(reward)
        self.weights[i] *= np.exp(-2 * self.__gamma__() * loss)
        self.t += 1
