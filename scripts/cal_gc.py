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

parser = argparse.ArgumentParser(description='calculate simulation acceptances')
parser.add_argument('path', nargs=1, help='path to yield results')
parser.add_argument('-a', default='acceptance.pkl', type=str, help='acceptance file', dest='acc_file')
parser.add_argument('-e', default=2.2, type=float, help='beam energy', dest='e')
# 2.2 gev: 735.34935, 1.1 gev: 177.34404
parser.add_argument('-l', default=735.34935, type=float, help='luminosity per file', dest='lumi_per_file')
parser.add_argument('-o', default='form_factors.pkl', type=str, help='output file', dest='output_file')
parser.add_argument('-r', default='rad_correction.dat', type=str, help='radiative correction file', dest='rad_cor_file')

args = vars(parser.parse_args())
path = args['path'][0]
ei = args['e']
lumi_per_file = args['lumi_per_file']
acceptance_file = args['acc_file']
rad_correction_file = args['rad_cor_file']
output_file = args['output_file']

if os.path.exists(output_file):
    os.remove(output_file)

bins = sim_configs.binning['bins']
low, high = sim_configs.binning['range']
bin_centers = numpy.linspace(low + (high - low) / bins / 2, high - (high - low) / bins / 2, bins)
bin_edges = numpy.linspace(low, high, bins + 1)

theta = bin_centers * numpy.pi / 180
theta_edges = bin_edges * numpy.pi / 180

result = {}

# load yield results
n_files = 0
yields = numpy.zeros(sim_configs.binning['bins'])
for filename in glob.glob(os.path.join(path, '*.pkl')):
    with open(filename, 'rb') as f:
        result = pickle.load(f)
    yields += result['hist_theta']
    n_files += 1
dyields = numpy.sqrt(yields)
lumi = n_files * lumi_per_file
result['yields'] = numpy.copy(yields)

# apply acceptance
with open(acceptance_file, 'rb') as f:
    acceptance_dict = pickle.load(f)
acc = acceptance_dict['acceptance']
systematics_of_acc = sim_configs.error_of_acceptance
dacc = numpy.sqrt((acceptance_dict['error_of_acceptance'] / acc)**2 + systematics_of_acc**2) * acc
yields = yields / acc
dyields = yields * numpy.sqrt((1 / dyields)**2 + (dacc / acc)**2)  # absolute error

# calculate q2
ef = get_ef(ei, theta, _m_d)
q2 = 4 * ei * ef * numpy.sin(theta / 2)**2
eta = q2 / (4 * _m2_d)
q2 = q2 * _gev_to_inv_fm**2
dq2 = numpy.zeros(q2.shape)

# convert to differential cross-section
omega = -2 * numpy.pi * numpy.diff(numpy.cos(theta_edges))
xs_raw = yields / lumi / omega
dxs_raw = dyields / lumi / omega

# corrections
rad_cor, drad_cor = sim_corrections.get_radiative_cor(rad_correction_file)
bin_center_cor, _ = sim_corrections.get_bin_center_cor(ei)
xs = xs_raw * rad_cor * bin_center_cor
systematics_of_xs = numpy.sqrt(sim_configs.error_of_event_selection**2 + sim_configs.error_of_other_sources**2)
dxs = numpy.sqrt((dxs_raw / xs_raw)**2 + (drad_cor / rad_cor)**2 + systematics_of_xs**2) * xs

# get structure function A
gc0, gm0, gq0 = form_factors.abbott_2000_1(q2)
mott = born_xs.mott(ei, theta) * ef / ei  # corrected mott xs
a = xs / mott - (4 / 3 * eta * (1 + eta) * gm0**2) * numpy.tan(theta / 2)
da = dxs / mott

# get form factor gc
gc = numpy.sqrt(a - 8 / 9 * (eta * gq0)**2 + 2 / 3 * eta * gm0**2)
dgc = da / (2 * gc)

print(gc, dgc)

result['Q2'] = q2
result['dQ2'] = dq2
result['GC'] = gc
result['dGC'] = dgc

with open(output_file, 'wb') as f:
    pickle.dump(result, f)
