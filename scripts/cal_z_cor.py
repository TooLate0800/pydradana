#!/usr/bin/env python3

import argparse
import glob
import os
import pickle

import numpy
from scipy.interpolate import interp1d

from pydradana import sim_configs

parser = argparse.ArgumentParser(description='calculate simulation z correction')
parser.add_argument('path', nargs=1, help='path to cooked acceptance simulation pkl files')
parser.add_argument('-o', default='z_correction.pkl', type=str, help='output filename', dest='output_file')

args = vars(parser.parse_args())
path = args['path'][0]
output_file = args['output_file']

if os.path.exists(output_file):
    os.remove(output_file)

n_files = 0
hist_z_sum = numpy.zeros(sim_configs.binning['bins'] + 4)
hist_z_count = numpy.zeros(sim_configs.binning['bins'] + 4)
for filename in glob.glob(os.path.join(path, '*.pkl')):
    with open(filename, 'rb') as f:
        result = pickle.load(f)
    hist_z_sum += result['hist_z_sum']
    hist_z_count += result['hist_z_count']
    n_files += 1

print('loaded {} files'.format(n_files))

averaged_z = hist_z_sum / hist_z_count

bins = sim_configs.binning['bins']
low, high = sim_configs.binning['range']
bin_size = (high - low) / bins
bin_centers = numpy.linspace(low - 2 * bin_size + 0.5 * bin_size, high + 2 * bin_size - 0.5 * bin_size, bins + 4)
z_correction = interp1d(bin_centers * numpy.pi / 180, averaged_z, kind='cubic')

result = {}
result['averaged_z'] = averaged_z
result['z_correction'] = z_correction

print(result)

with open(output_file, 'wb') as f:
    pickle.dump(result, f)
