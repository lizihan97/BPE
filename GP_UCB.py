import logging
import random

import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular


def random_argmax(arr):
    m = arr.max()
    idx = np.nonzero(arr == m)[0]
    return random.choice(idx)


def SE_kernel(l):
    return lambda a, b: np.exp(-np.square(np.linalg.norm(a - b)) / (2 * l ** 2))


def generate_domain(dimension, scale, size):
    meshgrid = np.array(np.meshgrid(*[np.linspace(-scale, scale, size) for _ in range(dimension)]))
    domain = meshgrid.reshape(dimension, -1).T
    return domain


class GP_UCB(object):
    def __init__(self, f, noise=0.02, dimension=2, scale=5, size=50, l=0.5,
                 beta=lambda i: np.log((2 * i) ** 3), reset=False):
        """
        :param f: the black box function to maximize
        :param noise: standard deviation of noise
        :param dimension: dimension of domain
        :param scale: scale of domain
        :param size: size of each axis
        :param l: param for SE_kernel
        """
        self.reset = reset
        self.M_i = generate_domain(dimension, scale, size)
        self.kernel = SE_kernel(l)
        self.noise = noise
        self.mu = np.array([0. for _ in range(self.M_i.shape[0])])
        self.sigma = np.array([1. for _ in range(self.M_i.shape[0])])
        self.observe = lambda x: f(x) + np.random.normal(0, self.noise)
        self.beta = beta
        self.f, self.X, self.Y = f, [], []
        self.fmax = f(self.M_i).max()
        self.K_inv, self.K_inv_y = None, None
        logging.info('noise = {}, dimension = {}, scale = {}, size = {}, fmax = {}, l = {}'
                     .format(noise, dimension, scale, size, self.fmax, l))
        logging.info('M_0 shape = {}'.format(self.M_i.shape))
        logging.info('mu_0 = {}'.format(self.mu))
        logging.info('sigma_0 = {}'.format(self.sigma))

    def update_K(self):
        K = np.array([[self.kernel(x1, x2) for x2 in self.X] for x1 in self.X]) \
            + np.eye(len(self.X)) * (self.noise ** 2)
        L = cholesky(K, lower=True)
        L_inv = solve_triangular(L.T, np.eye(L.shape[0]))
        self.K_inv = L_inv.dot(L_inv.T)
        self.K_inv_y = np.dot(self.K_inv, self.Y)

    def get_posterior(self, x, update_mean=False):
        k_t_x = [self.kernel(_, x) for _ in self.X]
        k_x = self.kernel(x, x)
        sigma = np.sqrt(k_x - np.dot(np.dot(k_t_x, self.K_inv), k_t_x))
        if update_mean:
            mu = np.dot(k_t_x, self.K_inv_y)
            return mu, sigma
        else:
            return None, sigma

    def update_posterior(self, x_t, update_mean=False):
        self.X.append(x_t)
        self.Y.append(self.observe(x_t.reshape(1, -1))[0])
        self.update_K()
        if update_mean:
            self.mu, self.sigma = np.array([self.get_posterior(_, True) for _ in self.M_i], dtype=float).T
        else:
            _, self.sigma = np.array([self.get_posterior(_) for _ in self.M_i], dtype=float).T

    def run(self, T, output):
        """
        :param T: time horizon
        """
        logging.info('GP_UCB, T = {}'.format(T))
        R_t = 0
        regret = []
        for t in range(1, T + 1):
            x_t = self.M_i[random_argmax(self.mu + np.sqrt(self.beta(t)) * self.sigma)]
            if self.reset:
                self.X, self.Y = [], []
            self.update_posterior(x_t, update_mean=True)
            r_t = self.fmax - self.f(x_t.reshape(1, -1)).reshape(1)[0]
            R_t += r_t
            regret.append([t, R_t])
            logging.info('t = {}, x_t = {}, r_t = {}, R_t = {};'
                         .format(t, x_t, r_t, R_t))
            print(t, r_t, R_t)
        pd.DataFrame(regret, columns=['t', 'R_t']).to_csv(output, header=True, index=False)
        x = self.M_i[random_argmax(self.mu)]
        logging.info('x = {}, f(x) = {}, fmax = {}'.format(x, self.f(x.reshape(1, -1)), self.fmax))
