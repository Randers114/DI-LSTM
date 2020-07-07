import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

mu = [10.481837, 31.22556,   8.130912]
sigma = [6.6800098e+00, 7.0718117e+00, 5.0141811e-03]
pi = [4.0720615e-01, 5.9279341e-01, 5.2170287e-07]

def mix_pdf(x, loc, scale, weights):
    d = np.zeros_like(x)
    for mu, sigma, pi in zip(loc, scale, weights):
        d += pi * norm.pdf(x, loc=mu, scale=sigma)
    return d






def plot_mixture_custom(mu, pi, sigma, color, plt=plt, show=True):
    grid = np.arange(0, 45, 0.01)
    plt.plot(grid, mix_pdf(grid, mu, sigma, pi), label='Almost', color=color)

    if(show):
        plt.show()
