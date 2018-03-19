#!/usr/bin/env python3

# For python 2-3 compatibility
from __future__ import division, print_function

import numpy
from scipy import integrate

from . import born_xs, sim_configs

__all__ = ['load_from_file', 'get_bin_center_cor']


def _get_xs_sin_func(xs_func, ei):

    def xs_sin_func(theta):
        return born_xs.ed(ei, theta) * numpy.sin(theta) * 2 * numpy.pi

    return xs_sin_func


def load_from_file(filename):
    ei, theta_min, theta_max, xs, dxs = numpy.loadtxt(filename, usecols=(0, 1, 2, 5, 6), unpack=True)

    theta_min = theta_min * numpy.pi / 180
    theta_max = theta_max * numpy.pi / 180
    omega = 2 * numpy.pi * (numpy.cos(theta_min) - numpy.cos(theta_max))

    born_xs_sin_func = _get_xs_sin_func(born_xs.ed, ei[0])

    xs_0, dxs_0 = numpy.array([integrate.quad(born_xs_sin_func, theta_min[i], theta_max[i]) for i in range(len(theta_min))]).T / omega

    cor = xs_0 / xs
    dcor = numpy.sqrt((dxs_0 / xs_0)**2 + (dxs / xs)**2) * cor

    return cor, dcor


def get_bin_center_cor(ei):
    bins = sim_configs.binning['bins']
    low, high = sim_configs.binning['range']
    bin_centers = numpy.linspace(low + (high - low) / bins / 2, high - (high - low) / bins / 2, bins)
    bin_edges = numpy.linspace(low, high, bins + 1)

    theta = bin_centers * numpy.pi / 180
    theta_edges = bin_edges * numpy.pi / 180

    omega = -2 * numpy.pi * numpy.diff(numpy.cos(theta_edges))

    born_xs_sin_func = _get_xs_sin_func(born_xs.ed, ei)

    xs_0, _ = numpy.array([integrate.quad(born_xs_sin_func, theta_edges[i], theta_edges[i + 1]) for i in range(len(theta_edges) - 1)]).T
    xs_0 = xs_0 / omega
    xs_0_center = born_xs.ed(ei, theta)

    cor = xs_0_center / xs_0

    return cor
