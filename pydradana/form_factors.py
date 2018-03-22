#!/usr/bin/env python3

# For python 2-3 compatibility
from __future__ import division, print_function

import numpy
from scipy import constants

__all__ = ['monopole', 'dipole', 'gaussian', 'arrington_2004', 'kelly_2004', 'venkat_2011', 'abbott_2000_1', 'abbott_2000_2']

# Constants
_mu_p = constants.value('proton mag. mom. to nuclear magneton ratio')
_m_p = constants.value('proton mass energy equivalent in MeV') * 1e-3
_m_d = constants.value('deuteron mass energy equivalent in MeV') * 1e-3
_inv_fm_to_gev = constants.hbar * constants.c / constants.e * 1e6  # fm^{-1} to GeV
_gev_to_inv_fm = 1 / _inv_fm_to_gev


def monopole(q2, r, *args, **kwargs):
    gc = 1 / (1 + q2 * r**2 / 6)
    gm = 0
    gq = 0
    return gc, gm, gq


def dipole(q2, r, *args, **kwargs):
    gc = 1 / (1 + q2 * r**2 / 12)**2
    gm = 0
    gq = 0
    return gc, gm, gq


def gaussian(q2, r, *args, **kwargs):
    gc = 1 * numpy.exp(-q2 * r**2 / 6)
    gm = 0
    gq = 0
    return gc, gm, gq


# Proton form factors
def arrington_2004(q2, *args, **kwargs):
    # J. Arrington, Phys. Rev. C 69(2004)022201
    q2_gev = q2 * _inv_fm_to_gev**2
    ae = [0, 3.226, 1.508, -0.3773, 0.611, -0.1853, 1.596e-2]
    am = [0, 3.19, 1.355, 0.151, -1.14e-2, 5.33e-4, -9.00e-6]
    ge = 1 / (1 + sum([ae[i] * q2_gev**i for i in range(1, 7)]))
    gm = _mu_p / (1 + sum([am[i] * q2_gev**i for i in range(1, 7)]))
    return ge, gm, 0


def kelly_2004(q2, *args, **kwargs):
    # J. J. Kelly, Phys. Rev. C 70(2004)068202
    tau = q2 * _inv_fm_to_gev**2 / (4 * _m_p**2)
    ae = [0, -0.24]
    be = [0, 10.98, 12.82, 21.97]
    am = [0, 0.12]
    bm = [0, 10.97, 18.86, 6.55]
    ge = (1 + tau * ae[1]) / (1 + sum([be[i] * tau**i for i in range(1, 4)]))
    gm = _mu_p * (1 + tau * am[1]) / (1 + sum([bm[i] * tau**i for i in range(1, 4)]))
    return ge, gm, 0


def venkat_2011(q2, *args, **kwargs):
    # S. Venkat, J. Arrington, G. A. Miller, and X. Zhan, Phys. Rev. C 83(2011)015203
    tau = q2 * _inv_fm_to_gev**2 / (4 * _m_p**2)
    ae = [0, 2.90966, -1.11542229, 3.866171e-2]
    be = [0, 14.5187212, 40.88333, 99.999998, 4.579e-5, 10.3580447]
    am = [0, -1.43573, 1.19052066, 2.5455841e-1]
    bm = [0, 9.70703681, 3.7357e-4, 6.0e-8, 9.9527277, 12.7977739]
    ge = (1 + sum([ae[i] * tau**i for i in range(1, 4)])) / (1 + sum([be[i] * tau**i for i in range(1, 6)]))
    gm = _mu_p * (1 + sum([am[i] * tau**i for i in range(1, 4)])) / (1 + sum([bm[i] * tau**i for i in range(1, 6)]))
    return ge, gm, 0


#Deuteron form factors
def abbott_2000_1(q2, *args, **kwargs):
    # Parameterization I in Eur. Phys. J A 7(2000)421
    # r_0 = 2.094
    gc0, qc0 = 1, 4.21
    gm0, qm0 = 1.714, 7.37
    gq0, qq0 = 25.83, 8.1
    ac = [0, 6.740e-1, 2.246e-2, 9.806e-3, -2.709e-4, 3.793e-6]
    am = [0, 5.804e-1, 8.701e-2, -3.624e-3, 3.448e-4, -2.818e-6]
    aq = [0, 8.796e-1, -5.656e-2, 1.933e-2, -6.734e-4, 9.438e-6]
    gc = gc0 * (1 - q2 / qc0**2) / (1 + sum([ac[i] * q2**i for i in range(1, 6)]))
    gm = gm0 * (1 - q2 / qm0**2) / (1 + sum([am[i] * q2**i for i in range(1, 6)]))
    gq = gq0 * (1 - q2 / qq0**2) / (1 + sum([aq[i] * q2**i for i in range(1, 6)]))
    return gc, gm, gq


def abbott_2000_2(q2, *args, **kwargs):
    # Parameterization II in Eur. Phys. J A 7(2000)421
    # r_0 = 2.088
    eta = q2 / (4 * (_m_d * _gev_to_inv_fm)**2)
    delta = (0.89852 * _gev_to_inv_fm)**2
    gq2 = 1 / (1 + q2 / (4 * delta))**2
    a = [1.57057, 12.23792, -42.04576, 27.92014]
    alpha = [1.52501, 8.75139, 15.97777, 23.20415]
    b = [0.07043, 0.14443, -0.27343, 0.05856]
    beta = [43.67795, 30.05435, 16.43075, 2.80716]
    c = [-0.16577, 0.27557, -0.05382, -0.05598]
    gamma = [1.87055, 14.95683, 28.04312, 41.12940]
    g0 = sum([a[i] / (alpha[i] + q2) for i in range(4)])
    g1 = numpy.sqrt(q2) * sum([b[i] / (beta[i] + q2) for i in range(4)])
    g2 = q2 * sum([c[i] / (gamma[i] + q2) for i in range(4)])
    C = gq2**2 / (2 * eta + 1)
    sqrt_2_eta = numpy.sqrt(2 * eta)
    gc = C * ((1 - 2 / 3 * eta) * g0 + 8 / 3 * sqrt_2_eta * g1 + 2 / 3 * (2 * eta - 1) * g2)
    gm = C * (2 * g0 + 2 * (2 * eta - 1) / sqrt_2_eta * g1 - 2 * g2)
    gq = C * (-1 * g0 + 2 / sqrt_2_eta * g1 - (1 + 1 / eta) * g2)
    return gc, gm, gq
