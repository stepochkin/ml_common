import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal


class GMM(object):
    def __init__(self, x, mu_init, c_init, pi_init, n_mixtures):
        self.x = np.vstack(x)  # Inputs always vertically stacked (more convenient)
        self.mu = mu_init  # Initial means of Gaussian mixture
        self.c = c_init  # Initial covariance matrices of Gaussian mixture
        self.pi = pi_init  # Initial mixture proportions of Gaussian mixgure
        self.n_mixtures = n_mixtures  # Number of components in mixture
        self.n, self.d = np.shape(self.x)  # Number of data points and dimension of problem
        self.ez = None

    def expectation(self):
        for n in range(self.n):
            den = 0.0
            for k in range(self.n_mixtures):
                den += self.pi[k] * multivariate_normal.pdf(self.x[n], self.mu[k], self.c[k])
            for k in range(self.n_mixtures):
                num = self.pi[k] * multivariate_normal.pdf(self.x[n], self.mu[k], self.c[k])
                self.ez[n, k] = num / den

    def maximisation(self, x, labels):
        for k in range(self.n_mixtures):
            nk = np.sum(labels[:, k])
            self.pi[k] = nk / self.n

            # Note - should vectorize this next bit in the future as it will be a lot faster
            self.mu[k] = 0.0
            for n in range(self.n):
                self.mu[k] += 1 / nk * labels[n, k] * x[n]
            self.c[k] = np.zeros([self.n_mixtures, self.n_mixtures])
            for n in range(self.n):
                self.c[k] += 1 / nk * labels[n, k] * np.vstack(x[n] - self.mu[k]) * (x[n] - self.mu[k])

    def train(self, ni):
        print('Training...')
        for i in range(ni):
            print('Iteration', i)
            self.expectation()
            self.maximisation(self.x, self.ez)

    def plot(self):
        if self.d == 2:
            # Plot contours
            r1 = np.linspace(-5, 5, 100)
            r2 = np.linspace(-5, 5, 100)
            x_r1, x_r2 = np.meshgrid(r1, r2)
            pos = np.empty(x_r1.shape + (2,))
            pos[:, :, 0] = x_r1
            pos[:, :, 1] = x_r2
            for k in range(self.n_mixtures):
                p = multivariate_normal(self.mu[k], self.c[k])
                plt.contour(x_r1, x_r2, p.pdf(pos))
            plt.show()
        else:
            print('Currently only produce plots for 2D problems.')
