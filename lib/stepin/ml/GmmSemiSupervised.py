import numpy as np

from stepin.ml import GMM


class GmmSemiSupervised(GMM):
    def __init__(self, x, x_labelled, y, mu_init, c_init, pi_init, n_mixtures):
        super().__init__(x, mu_init, c_init, pi_init, n_mixtures)
        self.x_labelled = np.vstack(x_labelled)
        self.x_all = np.vstack((self.x_labelled, self.x))
        self.y = np.vstack(y)
        self.n_labelled, self.d = np.shape(self.x_labelled)
        self.ez = np.zeros([self.n, n_mixtures])  # Initialise expected labels

    def train(self, ni):
        print('Training...')
        for i in range(ni):
            print('Iteration', i)
            self.expectation()
            labels = np.vstack((self.y, self.ez))
            self.maximisation(self.x_all, labels)
