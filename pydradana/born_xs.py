#!/usr/bin/env python3

# For python 2-3 compatibility
from __future__ import division, print_function

import math

import numpy
from scipy import constants

from . import _form_factors

__all__ = ['get_ef', 'mott', 'ee', 'ep', 'ed']

_m_e = constants.value("electron mass energy equivalent in MeV") * 1e-3
_m_p = constants.value("proton mass energy equivalent in MeV") * 1e-3
_m_d = constants.value("deuteron mass energy equivalent in MeV") * 1e-3
_m2_e = _m_e**2
_m2_p = _m_p**2
_m2_d = _m_d**2
_alpha = constants.alpha
_inv_fm_to_gev = constants.hbar * constants.c / constants.e * 1e6  # fm^{-1} to GeV
_inv_gev_to_fm = _inv_fm_to_gev
_gev_to_inv_fm = 1 / _inv_fm_to_gev
_inv_gev_to_mkb = _inv_gev_to_fm**2 * 1e4  # GeV^{-2} to microbarn


def get_ef(ei_e, theta_e, m_h):
    sin_theta = numpy.sin(theta_e)
    cos_theta = numpy.cos(theta_e)
    return ((ei_e + m_h) * (ei_e * m_h + _m2_e) + numpy.sqrt(m_h**2 - _m2_e * sin_theta**2) *
            (ei_e**2 - _m2_e) * cos_theta) / ((ei_e + m_h)**2 - (ei_e**2 - _m2_e) * cos_theta**2)


def mott(ei_e, theta_e):
    cos_theta_2 = numpy.cos(theta_e / 2)
    sin2_theta_2 = 1 - cos_theta_2**2
    return (_alpha * cos_theta_2 / (2 * ei_e * sin2_theta_2))**2 * _inv_gev_to_mkb


def ee(ei_1, theta_1):
    # sin_theta = numpy.sin(theta_1)
    cos_theta = numpy.cos(theta_1)
    C = (ei_1 - _m_e) * cos_theta**2
    ef_1 = _m_e * (ei_1 + _m_e + C) / (ei_1 + _m_e - C)

    pi_1 = math.sqrt(ei_1**2 - _m2_e)
    pf_1 = numpy.sqrt(ef_1**2 - _m2_e)

    # vi_1 = [0, 0, pi_1, ei_1]
    # vi_2 = [0, 0, 0, _m_e]
    # vf_1 = [pf_1 * sin_theta, 0, pf_1 * cos_theta, ef_1]
    # s = (ei_1 + _m_e)**2 - pi_1**2  # (vi_1 + vi_2) * (vi_1 + vi_2)
    # t = (ef_1 - ei_1)**2 - (pf_1 * sin_theta)**2 - (pf_1 * cos_theta - pi_1)**2  # (vf_1 - vi_1) * (vf_1 - vi_1)
    # u = (ef_1 - _m_e)**2 - (pf_1 * sin_theta)**2 - (pf_1 * cos_theta)**2  # (vf_1 - vi_2) * (vf_1 - vi_2)
    s = 2 * _m_e * (_m_e + ei_1)
    t = 2 * (_m2_e - ei_1 * ef_1 + pi_1 * pf_1 * cos_theta)
    u = 2 * _m_e * (_m_e - ef_1)

    a_dir = (s - 2 * _m2_e)**2 + (u - 2 * _m2_e)**2 + 4 * _m2_e * t
    a_ex = (s - 2 * _m2_e)**2 + (t - 2 * _m2_e)**2 + 4 * _m2_e * u
    a_int = -(s - 2 * _m2_e) * (s - 6 * _m2_e)

    return 2 * _alpha**2 * (cos_theta / (ei_1 + _m_e - C)) * (a_dir / t**2 + a_ex / u**2 - 2 * a_int / (t * u)) * _inv_gev_to_mkb


def ep(ei_e, theta_e):
    ef_e = get_ef(ei_e, theta_e, _m_p)
    qq = 2 * _m_p * (ef_e - ei_e)
    tau = -qq / (4 * _m2_p)
    eps = 1 / (1 - 2 * (1 + tau) * (qq + 2 * _m2_e) / (4 * ei_e * ef_e + qq))  # modified epsilon
    dd = (ef_e / ei_e) * numpy.sqrt((ei_e**2 - _m2_e) / (ef_e**2 - _m2_e))

    # the lepton mass isn't neglected here, see arXiv:1401.2959
    mott_ep = (_alpha / (2 * ei_e))**2 * ((1 + qq / (4 * ei_e * ef_e)) / (qq / (4 * ei_e * ef_e))**2) * (1 / dd)
    mott_ep = mott_ep * (_m_p * (ef_e**2 - _m2_e) / (_m_p * ei_e * ef_e + _m2_e * (ef_e - ei_e - _m_p))) * _inv_gev_to_mkb

    ge, gm, _ = _form_factors.venkat_2011(-qq * _gev_to_inv_fm**2)

    return mott_ep * (1 / (eps * (1 + tau))) * (eps * ge**2 + tau * gm**2)


def ed(ei_e, theta_e):
    sin_theta_2 = numpy.sin(theta_e / 2)
    tan_theta_2 = numpy.tan(theta_e / 2)
    ef_e = get_ef(ei_e, theta_e, _m_d)
    q2 = 4 * ei_e * ef_e * sin_theta_2**2
    eta = q2 / (4 * _m2_d)

    gc, gm, gq = _form_factors.abbott_2000_1(q2 * _gev_to_inv_fm**2)

    A = gc**2 + 8 / 9 * (eta * gq)**2 + 2 / 3 * eta * gm**2
    B = 4 / 3 * eta * (1 + eta) * gm**2

    return mott(ei_e, theta_e) * ef_e / ei_e * (A + B * tan_theta_2**2)
