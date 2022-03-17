import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


def SE_kernel(l):
    return lambda a, b: np.exp(-np.square(np.linalg.norm(a - b)) / (2 * l ** 2))


def generate_domain(dimension, scale, size):
    meshgrid = np.array(np.meshgrid(*[np.linspace(-scale, scale, size) for _ in range(dimension)]))
    domain = meshgrid.reshape(dimension, -1).T
    return domain


def generate_gp(size, dimension, scale, l, name):
    X = generate_domain(dimension, scale, size)
    kernel = SE_kernel(l)
    k = np.array([[kernel(x1, x2) for x2 in X] for x1 in X])
    y = np.random.multivariate_normal(mean=np.zeros(size ** dimension), cov=k)

    # save values
    df = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
    df.columns = ['x_1', 'x_2', 'y']
    df.to_csv('{}.csv'.format(name), header=True, index=False)
    points = pd.read_csv('{}.csv'.format(name))
    gpr = GaussianProcessRegressor(kernel=RBF(l), random_state=0)
    # gpr = GaussianProcessRegressor(kernel=RBF(l, length_scale_bounds='fixed'), random_state=0)
    gpr.fit(points[['x_1', 'x_2']], points[['y']])

    meshgrid = np.array(np.meshgrid(np.linspace(-scale, scale, 50),
                                    np.linspace(-scale, scale, 50)),
                        np.float64)
    Y = gpr.predict(meshgrid.reshape(2, -1).T) \
        .reshape(50, 50)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(meshgrid[0], meshgrid[1], Y, alpha=0.6)
    plt.show()


generate_gp(5, 2, 5, 0.5, 'f1')
