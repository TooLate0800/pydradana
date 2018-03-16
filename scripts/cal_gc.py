#!/usr/bin/env python3

import argparse
import glob
import os
import pickle

import numpy
from scipy import constants

from pydradana import sim_configs, born_xs
from pydradana.born_xs import get_ef

_m_d = constants.value('deuteron mass energy equivalent in MeV') * 1e-3

parser = argparse.ArgumentParser(description='calculate simulation acceptances')
parser.add_argument('path', nargs=1, help='path to yield results')
parser.add_argument('-a', default='acceptance.pkl', type=str, help='acceptance file', dest='acc_file')
parser.add_argument('-e', default=2.2, type=float, help='beam energy', dest='e')
# 2.2 gev: 735.34935, 1.1 gev:
parser.add_argument('-l', default=735.34935, type=float, help='luminosity per file', dest='lumi_per_file')

args = vars(parser.parse_args())
path = args['path'][0]
ei = args['e']
acceptance_file = args['acc_file']
lumi_per_file = args['lumi_per_file']

if os.path.exists(os.path.join(path, 'gc.pkl')):
    os.remove(os.path.join(path, 'gc.pkl'))

with open(acceptance_file, 'rb') as f:
    acceptance_dict = pickle.load(f)

acc = acceptance_dict['acceptance']
dacc = acceptance_dict['error_of_acceptance']

n_files = 0
yields = numpy.zeros(sim_configs.binning['bins'])
for filename in glob.glob(os.path.join(path, '*.pkl')):
    with open(filename, 'rb') as f:
        result = pickle.load(f)
    yields += result['hist_theta']
    n_files += 1

lumi = n_files * lumi_per_file

bins = sim_configs.binning['bins']
low, high = sim_configs.binning['range']
bin_centers = numpy.linspace(low + (high - low) / bins / 2, high - (high - low) / bins / 2, bins)
bin_edges = numpy.linspace(low, high, bins + 1)

theta = bin_centers / 180 * numpy.pi
theta_edges = bin_edges / 180 * numpy.pi

dyields = numpy.sqrt(yields)

yields = yields / acc
dyields = yields * numpy.sqrt((1 / dyields)**2 + (dacc / acc)**2)  # absolute error

ef = get_ef(ei, theta, _m_d)
q2 = 4 * ei * ef * numpy.sin(theta / 2)**2
dq2 = numpy.zeros(q2.shape)

omega = -2 * numpy.pi * numpy.diff(numpy.cos(theta_edges))

xs_raw = yields / lumi / omega
dxs_raw = dyields / lumi / omega

# no correction at this moment
xs = xs_raw
dxs = dxs_raw

mott = born_xs.mott(ei, theta) * ef / ei  # corrected mott xs

a = xs / mott
da = dxs / mott

gc = numpy.sqrt(a)
dgc = da / (2 * numpy.sqrt(a))

result = {}
result['Q2'] = q2
result['dQ2'] = dq2
result['GC'] = gc
result['dGC'] = dgc

with open(os.path.join(path, 'gc.pkl'), 'wb') as f:
    pickle.dump(result, f)
