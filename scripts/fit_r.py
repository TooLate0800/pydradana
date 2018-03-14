#!/usr/bin/env python3

from __future__ import division, print_function

import multiprocessing
import os
from os.path import exists, join

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import numpy
from scipy.optimize import curve_fit

from pydradana import RFitter

output_path = '.'


def gaus(x, C, mu, sigma):
    return C * numpy.exp(-(x - mu)**2 / (2 * sigma**2))


def do_fit(N=100000, model_gen='dipole', model_fit='dipole', r0=2.130, lock=None):
    print('generation model = {}, fit model = {}'.format(model_gen, model_fit))

    result = numpy.empty(N, dtype=float)

    fitter = RFitter()
    fitter.load_data()
    fitter.set_range(0, 2)

    for i in range(1, N + 1):
        fitter.gen_model(model=model_gen, r0=r0)
        fitter.add_noise(model='gaussian')

        r, _ = fitter.fit(model=model_fit, method='least_squares', r0=r0)

        result[i - 1] = r

    r_mean = numpy.mean(result[(result < r0 * 1.15) & (result > r0 * 0.85)])
    r_std = numpy.std(result[(result < r0 * 1.15) & (result > r0 * 0.85)])

    r_u = r_mean + 4 * r_std
    r_d = r_mean - 4 * r_std

    hist, bin_edges = numpy.histogram(result, bins=200, range=(r_d, r_u))
    x = [(bin_edges[i] + bin_edges[i + 1]) / 2.0 for i in range(len(bin_edges) - 1)]

    try:
        popt, _ = curve_fit(gaus, x, hist, p0=[N / 200, r_mean, 0.1], bounds=([0, r_d, 0], [numpy.inf, r_u, numpy.inf]))
    except (ValueError, RuntimeError):
        popt = None

    model_fit_string = model_fit if not isinstance(model_fit, tuple) else '-'.join(map(str, model_fit))
    if lock is not None:
        with lock:
            with open(join(output_path, 'result.dat'), mode='a+') as fo:
                fo.write('{:<19}  {:>9}  {:.10f}  {:.10f}\n'.format(model_gen, model_fit_string, r_mean, r_std))

    print('{:<19}  {:<9}  {:.10f}  {:.10f}'.format(model_gen, model_fit_string, r_mean, r_std))

    font = {'size': 12}

    fig = plt.figure()
    ax = plt.gca()
    plt.xlabel(r'$r/fm$', fontdict=font)
    plt.hist(result, bins=200, range=(r_d, r_u), histtype='step')
    if popt is not None:
        plt.plot(x, gaus(x, *popt), label='fit')
    plt.text(0.1, 0.8, r'$r\,={:6.4f}$'.format(r_mean) + '\n' + r'$\sigma={:6.4f}$'.format(r_std), transform=ax.transAxes, fontdict=font)
    fig.savefig(join(output_path, '{}-{}.pdf'.format(model_gen, model_fit_string)))
    plt.close()


if exists(join(output_path, 'result.dat')):
    os.remove(join(output_path, 'result.dat'))

list_model_gen = ['dipole', 'monopole', 'gaussian', 'Abbott-2000-1', 'Abbott-2000-2']
list_model_fit = ['dipole', ('poly', 2), ('poly', 3), ('ratio', 1, 1), ('ratio', 1, 2), ('ratio', 2, 1), ('cf', 2), ('cf', 3)]

l = multiprocessing.Manager().Lock()
pool = multiprocessing.Pool()

for f in list_model_fit:
    for g in list_model_gen:
        pool.apply_async(do_fit, args=(100000, g, f, 2.130, l))

pool.close()
pool.join()
