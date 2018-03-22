#!/usr/bin/env python3

# For python 2-3 compatibility
from __future__ import division, print_function

import numpy
from scipy import integrate

from . import born_xs, sim_configs

__all__ = ['get_bin_center_cor', 'get_radiative_cor']


def _get_xs_sin_func(xs_func, ei):

    def xs_sin_func(theta):
        return born_xs.ed(ei, theta) * numpy.sin(theta) * 2 * numpy.pi

    return xs_sin_func


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
    dcor = numpy.zeros(cor.shape)

    return cor, dcor


def get_integrated_born_xs(ei, bin_edges):
    omega = -2 * numpy.pi * numpy.diff(numpy.cos(bin_edges))

    born_xs_sin_func = _get_xs_sin_func(born_xs.ed, ei)

    xs_0, dxs_0 = numpy.array([integrate.quad(born_xs_sin_func, bin_edges[i], bin_edges[i + 1]) for i in range(len(bin_edges) - 1)]).T / omega

    return xs_0, dxs_0
