#!/usr/bin/env python3

import numpy

from . import sim_configs
from .sim_reader import SimReader

__all__ = ['get_acceptance', 'get_yield']

# configs read from _sim_configs
_electron = sim_configs.electron
_proton = sim_configs.proton
_deuteron = sim_configs.deuteron

_z_center = sim_configs.target_center  # mm
_gem_res = sim_configs.gem_resolution

_binning = sim_configs.binning


def _get_general_cuts(reader):
    # find hits on each detector
    found_rd_0, found_rd_1 = reader.find_hits('RD', det_type='tracking', n_copy=2, pid=_deuteron)
    found_gem_0, found_gem_0_1, found_gem_1, found_gem_1_1 = reader.find_hits('GEM', det_type='tracking', n_copy=4, pid=_electron)
    found_sp = reader.find_hits('SP', pid=_electron)
    found_hc = reader.find_hits('HC', pid=_electron)

    found_gem_0[found_gem_0 < 0] = found_gem_0_1[found_gem_0 < 0]
    found_gem_1[found_gem_1 < 0] = found_gem_1_1[found_gem_1 < 0]

    is_good = (found_rd_0 >= 0) & (found_gem_0 >= 0) & (found_gem_1 >= 0) & (found_sp >= 0) & (found_hc >= 0)

    return found_rd_0, found_rd_1, found_gem_0, found_gem_1, found_sp, found_hc, is_good


def get_acceptance(filename, start, stop):
    r = SimReader(filename, start, stop)

    *_, is_good = _get_general_cuts(r)

    theta = r.GUN.Theta.get_column(0) * 180.0 / numpy.pi
    z = r.GUN.Z.get_column(0) - _z_center
    theta_good = theta[is_good]
    z_good = z[is_good]

    hist_theta, _ = numpy.histogram(theta, **_binning)
    hist_theta_good, _ = numpy.histogram(theta_good, **_binning)
    hist_z_good, _ = numpy.histogram(theta_good, weights=z_good, **_binning)

    result = {}
    result['hist_theta'] = hist_theta
    result['hist_theta_good'] = hist_theta_good
    result['hist_z_good'] = hist_z_good

    return result


def get_yield(filename, start, stop, z_correction=None):
    r = SimReader(filename, start, stop)

    _, _, found_gem_0, found_gem_1, _, _, is_good = _get_general_cuts(r)

    n_good = len(is_good[is_good])

    # silicon detector (1st layer)
    # x_rd_0 = r.RD.X.content[found_rd_0][is_good]
    # y_rd_1 = r.RD.Y.content[found_rd_1][is_good]

    # GEMs
    x_gem_0 = r.GEM.X.content[found_gem_0][is_good] + numpy.random.normal(0, _gem_res, n_good)
    y_gem_0 = r.GEM.Y.content[found_gem_0][is_good] + numpy.random.normal(0, _gem_res, n_good)
    z_gem_0 = r.GEM.Z.content[found_gem_0][is_good] - _z_center
    x_gem_1 = r.GEM.X.content[found_gem_1][is_good] + numpy.random.normal(0, _gem_res, n_good)
    y_gem_1 = r.GEM.Y.content[found_gem_1][is_good] + numpy.random.normal(0, _gem_res, n_good)
    z_gem_1 = r.GEM.Z.content[found_gem_1][is_good] - _z_center

    # HC
    # p_hc = r.HC.P.content[found_hc][is_good]

    #
    r_gem_0 = numpy.sqrt(x_gem_0**2 + y_gem_0**2)
    r_gem_1 = numpy.sqrt(x_gem_1**2 + y_gem_1**2)
    theta_gem_0 = numpy.arctan(r_gem_0 / z_gem_0)
    theta_gem_1 = numpy.arctan(r_gem_1 / z_gem_1)
    theta = (theta_gem_0 + theta_gem_1) / 2

    if z_correction is not None:
        dz = z_correction[0] * theta + z_correction[1]
        theta_gem_0 = numpy.arctan(r_gem_0 / (z_gem_0 - dz))
        theta_gem_1 = numpy.arctan(r_gem_1 / (z_gem_1 - dz))
        theta = (theta_gem_0 + theta_gem_1) / 2

    hist_theta, _ = numpy.histogram(theta, **_binning)

    result = {}
    result['hist_theta'] = hist_theta

    return result
