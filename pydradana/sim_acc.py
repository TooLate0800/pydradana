#!/usr/bin/env python3

import math

import numpy

from . import _sim_configs
from .sim_reader import SimReader

__all__ = ['process']

# configs read from _sim_configs
_electron = _sim_configs.electron
_proton = _sim_configs.proton
_deuteron = _sim_configs.deuteron
_z_offset = _sim_configs.target_offset  # mm
_binning = _sim_configs.binning


def process(filename, start, stop):
    r = SimReader(filename, start, stop)

    # find hits on each detector
    found_rd_0, _ = r.find_hits('RD', det_type='tracking', n_copy=2, pid=_deuteron)
    found_gem_0, found_gem_0_1, found_gem_1, found_gem_1_1 = r.find_hits('GEM', det_type='tracking', n_copy=4, pid=_electron)
    found_gem_0[found_gem_0 < 0] = found_gem_0_1[found_gem_0 < 0]
    found_gem_1[found_gem_1 < 0] = found_gem_1_1[found_gem_1 < 0]
    found_sp = r.find_hits('SP', pid=_electron)
    found_hc = r.find_hits('HC', pid=_electron)

    is_good = (found_rd_0 >= 0) & (found_gem_0 >= 0) & (found_gem_1 >= 0) & (found_sp >= 0) & (found_hc >= 0)

    theta = r.GUN.Theta.get_column(0) * 180.0 / math.pi
    z = r.GUN.Z.get_column(0) + _z_offset
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
