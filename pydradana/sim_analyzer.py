# Author: Chao Gu, 2018

# For python 2-3 compatibility
from __future__ import division, print_function

import numpy
from scipy import integrate

from . import born_xs, form_factors, sim_configs
from .sim_configs import m_d as _m_d
from .sim_configs import gev_to_inv_fm as _gev_to_inv_fm

__all__ = ['get_bin_center_correction', 'get_integrated_born_xs']

_bin_centers, _bin_edges = sim_configs.create_bins()
_theta = _bin_centers * numpy.pi / 180
_theta_edges = _bin_edges * numpy.pi / 180

_omega = -2 * numpy.pi * numpy.diff(numpy.cos(_theta_edges))


def _get_xs_sin_func(xs_func, ei):

    def xs_sin_func(theta):
        return born_xs.ed(ei, theta) * numpy.sin(theta) * 2 * numpy.pi

    return xs_sin_func


def get_integrated_born_xs(ei):
    born_xs_sin_func = _get_xs_sin_func(born_xs.ed, ei)

    xs_0, dxs_0 = numpy.array([integrate.quad(born_xs_sin_func, low, high) for low, high in zip(_theta_edges, _theta_edges[1:])]).T
    xs_0 = xs_0 / _omega
    dxs_0 = dxs_0 / _omega

    return xs_0, dxs_0


def get_bin_center_correction(ei):
    xs_0, _ = get_integrated_born_xs(ei)
    xs_0_center = born_xs.ed(ei, _theta)

    cor = xs_0_center - xs_0
    dcor = numpy.zeros(cor.shape)

    return cor, dcor


def calculate_gc(ei, lumi, yield_, dyield_, rad_cor, drad_cor, model='Abbott-2000-1'):
    # calculate q2 in fm
    ef = born_xs.get_ef(ei, _theta, _m_d)
    q2 = 4 * ei * ef * numpy.sin(_theta / 2)**2
    eta = q2 / (4 * _m_d**2)
    q2 = q2 * _gev_to_inv_fm**2
    dq2 = numpy.zeros(q2.shape)

    # convert to differential cross-section
    xs_raw = yield_ / lumi / _omega
    dxs_raw = dyield_ / lumi / _omega

    # apply radiative correction and bin center correction
    bin_center_cor, _ = get_bin_center_correction(ei)
    xs = xs_raw * rad_cor + bin_center_cor
    dxs = numpy.sqrt((dxs_raw / xs_raw)**2 + (drad_cor / rad_cor)**2) * xs

    # add systematics
    systematics = numpy.sqrt(sim_configs.error_of_acceptance**2 + sim_configs.error_of_detector**2 + sim_configs.error_of_event_selection**2)
    dxs = numpy.sqrt((dxs / xs)**2 + systematics**2) * xs

    # get structure function A
    mott = born_xs.mott(ei, _theta) * ef / ei  # corrected mott xs

    model_func = {
        'monopole': form_factors.monopole,
        'dipole': form_factors.dipole,
        'gaussian': form_factors.gaussian,
        'Abbott-2000-1': form_factors.abbott_2000_1,
        'Abbott-2000-2': form_factors.abbott_2000_2,
    }.get(model, None)
    _, gm_0, gq_0 = model_func(q2, 2.130)

    a = xs / mott - (4 / 3 * eta * (1 + eta) * gm_0**2) * numpy.tan(_theta / 2)**2
    da = dxs / mott

    # get form factor gc
    gc = numpy.sqrt(a - (8 / 9 * (eta * gq_0)**2 + 2 / 3 * eta * gm_0**2))
    dgc = da / (2 * gc)

    return q2, dq2, gc, dgc
