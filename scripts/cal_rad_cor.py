#!/usr/bin/env python3

import argparse
import glob
import os
import pickle

import numpy

from pydradana import sim_configs, sim_corrections

parser = argparse.ArgumentParser(description='calculate simulation radiative correction')
parser.add_argument('path', nargs=1, help='path to cooked radiative correction simulation pkl files')
parser.add_argument('-e', default=2.2, type=float, help='beam energy', dest='e')
# 2.2 gev: 735.34935, 1.1 gev: 177.34404
parser.add_argument('-l', default=735.34935, type=float, help='luminosity per file', dest='lumi_per_file')
parser.add_argument('-o', default='rad_correction.pkl', type=str, help='output filename', dest='output_file')

args = vars(parser.parse_args())
path = args['path'][0]
ei = args['e']
lumi_per_file = args['lumi_per_file']
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

print('loaded {} files'.format(n_files))

# convert to differential cross-section
omega = -2 * numpy.pi * numpy.diff(numpy.cos(theta_edges))
xs_sim = yields / lumi / omega
dxs_sim = dyields / lumi / omega

# born cross-section
xs_0, dxs_0 = sim_corrections.get_integrated_born_xs(ei, theta_edges)

rad_correction = xs_0 / xs_sim
systematics = sim_configs.error_of_radiative_correction
error_of_rad_correction = numpy.sqrt((dxs_sim / xs_sim)**2 + (dxs_0 / xs_0)**2 + systematics**2) * rad_correction

result = {}
result['rad_correction'] = rad_correction
result['error_of_rad_correction'] = error_of_rad_correction

print(result)

with open(output_file, 'wb') as f:
    pickle.dump(result, f)
