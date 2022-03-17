import datetime as dt
import logging
import os
import sys
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import BPE
import GP_UCB

now = dt.datetime.now().strftime("%Y%m%d%H%M%S")

BETA = {'2': lambda i: 2,
        '6': lambda i: 6,
        'log': lambda i: np.log((2 * i) ** 3)}


class function(object):
    def __init__(self, name, l):
        self.points = pd.read_csv('{}.csv'.format(name))
        self.gpr = GaussianProcessRegressor(kernel=RBF(l), random_state=0)
        # self.gpr = GaussianProcessRegressor(kernel=RBF(l, length_scale_bounds='fixed'), random_state=0)
        self.gpr.fit(self.points[['x_1', 'x_2']], self.points[['y']])

    def __call__(self, X):
        return self.gpr.predict(X)


if __name__ == '__main__':
    T = 1000
    f = sys.argv[1]  # f1 or f2
    beta_str = sys.argv[2]  # 2 or 6 or log
    beta = BETA[beta_str]
    reset_str = sys.argv[3]  # 0 or 1
    reset = bool(int(reset_str))  # 0 or 1
    print('reset = {}'.format(reset))
    DIR = os.getcwd()
    folder = '{}_{}_{}'.format(f, beta_str, reset_str)
    os.makedirs(folder)
    func = function(f, 0.5)
    for i in range(10):
        for B in [2, 3, 4, 6]:
            LOG_PATH = os.path.join(DIR, folder,
                                    'fixed_{}_{}_{}_{}_{}_{}_{}.log'.format(B, T, f, beta_str, reset_str, now, i))
            REGRET_PATH = os.path.join(DIR, folder,
                                       'fixed_{}_{}_{}_{}_{}_{}_{}.csv'.format(B, T, f, beta_str, reset_str, now, i))
            logging.basicConfig(filename=LOG_PATH, level=logging.INFO)
            logging.info('fixed, beta, reset = {}'.format(sys.argv[2:]))
            bo = BPE.BPE(func, l=0.5, beta=beta, reset=reset, fixed_batch_size=True)
            bo.run(B, T, REGRET_PATH)
    for i in range(10):
        # B = 0 for original BPE
        for B in [0, 2, 3, 4, 6]:
            LOG_PATH = os.path.join(DIR, folder,
                                    '{}_{}_{}_{}_{}_{}_{}.log'.format(B, T, f, beta_str, reset_str, now, i))
            REGRET_PATH = os.path.join(DIR, folder,
                                       '{}_{}_{}_{}_{}_{}_{}.csv'.format(B, T, f, beta_str, reset_str, now, i))
            logging.basicConfig(filename=LOG_PATH, level=logging.INFO)
            logging.info('beta, reset = {}'.format(sys.argv[2:]))
            bo = BPE.BPE(func, l=0.5, beta=beta, reset=reset, fixed_batch_size=False)
            bo.run(B, T, REGRET_PATH)
        """ GP_UCB """
        LOG_PATH = os.path.join(DIR, '{}_{}_{}_{}_{}.log'.format('GP_UCB', T, f, now, i))
        REGRET_PATH = os.path.join(DIR, '{}_{}_{}_{}_{}.csv'.format('GP_UCB', T, f, now, i))
        logging.basicConfig(filename=LOG_PATH, level=logging.INFO)
        bo = GP_UCB.GP_UCB(func, l=0.5, beta=beta, reset=False)
        bo.run(T, REGRET_PATH)
