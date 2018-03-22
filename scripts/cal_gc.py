#!/usr/bin/env python3

import argparse
import glob
import os
import pickle

import numpy
from scipy import constants

from pydradana import born_xs, form_factors, sim_configs, sim_corrections
from pydradana.born_xs import get_ef

_m_d = constants.value('deuteron mass energy equivalent in MeV') * 1e-3
_m2_d = _m_d**2
_inv_fm_to_gev = constants.hbar * constants.c / constants.e * 1e6  # fm^{-1} to GeV
_gev_to_inv_fm = 1 / _inv_fm_to_gev


def cal_gc(ei, y, dy, lumi, rad_cor, drad_cor):
    bins = sim_configs.binning['bins']
    low, high = sim_configs.binning['range']
    bin_centers = numpy.linspace(low + (high - low) / bins / 2, high - (high - low) / bins / 2, bins)
    bin_edges = numpy.linspace(low, high, bins + 1)

    theta = bin_centers * numpy.pi / 180
    theta_edges = bin_edges * numpy.pi / 180

    # calculate q2 in fm
    ef = get_ef(ei, theta, _m_d)
    q2 = 4 * ei * ef * numpy.sin(theta / 2)**2
    eta = q2 / (4 * _m2_d)
    q2 = q2 * _gev_to_inv_fm**2
    dq2 = numpy.zeros(q2.shape)

    # convert to differential cross-section
    omega = -2 * numpy.pi * numpy.diff(numpy.cos(theta_edges))
    xs_raw = y / lumi / omega
    dxs_raw = dy / lumi / omega

    # apply radiative correction
    bin_center_cor, _ = sim_corrections.get_bin_center_cor(ei)
    xs = xs_raw * rad_cor + bin_center_cor
    dxs = numpy.sqrt((dxs_raw / xs_raw)**2 + (drad_cor / rad_cor)**2) * xs

    # add systematics
    systematics = numpy.sqrt(sim_configs.error_of_acceptance**2 + sim_configs.error_of_detector**2 + sim_configs.error_of_event_selection**2)
    dxs = numpy.sqrt((dxs / xs)**2 + systematics**2) * xs

    # get structure function A
    _, gm_0, gq_0 = form_factors.abbott_2000_1(q2)
    mott = born_xs.mott(ei, theta) * ef / ei  # corrected mott xs
    a = xs / mott - (4 / 3 * eta * (1 + eta) * gm_0**2) * numpy.tan(theta / 2)**2
    da = dxs / mott

    # get form factor gc
    gc = numpy.sqrt(a - (8 / 9 * (eta * gq_0)**2 + 2 / 3 * eta * gm_0**2))
    dgc = da / (2 * gc)

    return q2, dq2, gc, dgc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calculate simulation acceptances')
    parser.add_argument('path', nargs=1, help='path to yield results')
    parser.add_argument('-e', default=2.2, type=float, help='beam energy', dest='e')
    # 2.2 gev: 735.34935, 1.1 gev: 177.34404
    parser.add_argument('-l', default=735.34935, type=float, help='luminosity per file', dest='lumi_per_file')
    parser.add_argument('-o', default='form_factors.pkl', type=str, help='output file', dest='output_file')
    parser.add_argument('-r', default='rad_correction.pkl', type=str, help='radiative correction file', dest='rad_cor_file')

    args = vars(parser.parse_args())
    path = args['path'][0]
    ei_ = args['e']
    lumi_per_file = args['lumi_per_file']
    rad_correction_file = args['rad_cor_file']
    output_file = args['output_file']

    if os.path.exists(output_file):
        os.remove(output_file)

    result = {}

    # load yield results
    n_files = 0
    yield_ = numpy.zeros(sim_configs.binning['bins'])
    for filename in glob.glob(os.path.join(path, '*.pkl')):
        with open(filename, 'rb') as f:
            result = pickle.load(f)
        yield_ += result['hist_theta']
        n_files += 1
    dyield_ = numpy.sqrt(yield_)
    result['yield'] = yield_
    result['dyield'] = dyield_

    lumi_ = n_files * lumi_per_file

    with open(rad_correction_file, 'rb') as f:
        rad_correction_dict = pickle.load(f)
    rad_cor_ = rad_correction_dict['rad_correction']
    drad_cor_ = rad_correction_dict['error_of_rad_correction']

    q2_, dq2_, gc_, dgc_ = cal_gc(ei_, yield_, dyield_, lumi_, rad_cor_, drad_cor_)

    result['Q2'] = q2_
    result['dQ2'] = dq2_
    result['GC'] = gc_
    result['dGC'] = dgc_

    with open(output_file, 'wb') as f:
        pickle.dump(result, f)
