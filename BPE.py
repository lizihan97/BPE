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


def get_batch_size(B, T, fixed_batch_size=False, c=0):
    """
    :param B: number of batches, 0 for Orig-BPE
    :param T: time horizon
    :param c: complexity of maximum information gain
    :return: array of length B+1 where the value at index i is the size of the i-th batch (index 0 is dummy)
    """
    if fixed_batch_size:
        batch_size = np.array([T/B for _ in range(B)]).astype(int)
        batch_size[-1] = T - sum(batch_size[:-1])
        return batch_size
    elif B == T:
        return np.ones(T).astype(int)
    elif B == 0:
        batch_size = [1]
        while True:
            N_i = np.ceil(np.sqrt(T*batch_size[-1]))
            if sum(batch_size) + N_i - 1 >= T:
                batch_size.append(T + 1 - sum(batch_size))
                break
            else:
                batch_size.append(N_i)
        return np.array(batch_size[1:]).astype(int)
    else:
        eta = (1 - c) / 2
        batch_size = []
        for i in range(1, B + 1):
            N_i = np.ceil(T ** ((1 - eta ** i) / (1 - eta ** B)))
            batch_size.append(N_i)
        s = sum(batch_size)
        batch_size = (np.array(batch_size) * T / s).astype(int)
        batch_size[-1] = T - sum(batch_size[:-1])
        return batch_size

class BPE(object):
    def __init__(self, f, noise=0.02, dimension=2, scale=5, size=50, l=0.5,
                 beta=lambda i: np.log((2 * i) ** 3), fixed_batch_size=False, reset=False):
        """
        :param f: the black box function to maximize
        :param noise: standard deviation of noise
        :param dimension: dimension of domain
        :param scale: scale of domain
        :param size: size of each axis
        :param l: param for SE_kernel
        :param beta: a function for i: beta
        :param reset: reset posterior at the start of each batch
        :param fixed_batch_size: use fixed batch size = T/B
        """
        self.fixed_batch_size = fixed_batch_size
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

    def update_M_i(self, i):
        if i == 1:
            pass
        else:
            max_lcb = (self.mu - np.sqrt(self.beta(i)) * self.sigma).max()
            M_i_idx = np.where(self.mu + np.sqrt(self.beta(i)) * self.sigma >= max_lcb)
            self.M_i = self.M_i[M_i_idx]
            self.mu = self.mu[M_i_idx]
            self.sigma = self.sigma[M_i_idx]
        logging.info('i = {}, M_i shape = {}, mu shape = {}, sigma shape = {}'
                     .format(i, self.M_i.shape, self.mu.shape, self.sigma.shape))

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

    def run(self, B, T, output):
        """
        :param B: number of batches, 0 for Orig-BPE
        :param T: time horizon
        """
        batch_size = get_batch_size(B, T, fixed_batch_size=self.fixed_batch_size)
        logging.info('B = {}, T = {}, N_i = {}'.format(B, T, batch_size))
        t = 1
        R_t = 0
        regret = []
        for i in range(0, len(batch_size)):
            self.update_M_i(i+1)
            if self.reset:
                self.X, self.Y = [], []
            for k in range(1, batch_size[i] + 1):
                x_t = self.M_i[random_argmax(self.sigma)]
                sigma_t = self.sigma.max()
                if k < batch_size[i]:
                    self.update_posterior(x_t)
                else:
                    self.update_posterior(x_t, update_mean=True)
                r_t = self.fmax - self.f(x_t.reshape(1, -1)).reshape(1)[0]
                R_t += r_t
                print(self.sigma.shape, self.sigma.min(), self.sigma.max(), R_t)
                regret.append([t, R_t])
                logging.info('i = {}, N_i = {}, k = {}, t = {}, x_t = {}, sigma_t = {}, r_t = {}, R_t = {};'
                             .format(i+1, batch_size[i], k, t, x_t, sigma_t, r_t, R_t))
                t += 1
        x = self.M_i[np.argmax(self.mu)]
        pd.DataFrame(regret, columns=['t', 'R_t']).to_csv(output, header=True, index=False)
        logging.info('x = {}, f(x) = {}, fmax = {}'.format(x, self.f(x.reshape(1, -1)), self.fmax))
