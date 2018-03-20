#!/usr/bin/env python3

import argparse
import glob
import os
import pickle

import numpy
from scipy.interpolate import interp1d

from pydradana import sim_configs

parser = argparse.ArgumentParser(description='calculate simulation acceptance')
parser.add_argument('path', nargs=1, help='path to cooked acceptance simulation pkl files')
parser.add_argument('-o', default='acceptance.pkl', type=str, help='output filename', dest='output_file')

args = vars(parser.parse_args())
path = args['path'][0]
output_file = args['output_file']

if os.path.exists(output_file):
    os.remove(output_file)

n_files = 0
hist_theta = numpy.zeros(sim_configs.binning['bins'])
hist_theta_good = numpy.zeros(sim_configs.binning['bins'])
for filename in glob.glob(os.path.join(path, '*.pkl')):
    with open(filename, 'rb') as f:
        result = pickle.load(f)
    hist_theta += result['hist_theta']
    hist_theta_good += result['hist_theta_good']
    n_files += 1

print('loaded {} files'.format(n_files))

acceptance = hist_theta_good / hist_theta
error_of_acceptance = numpy.sqrt(1 / hist_theta + 1 / hist_theta_good) * acceptance

result = {}
result['acceptance'] = acceptance
result['error_of_acceptance'] = error_of_acceptance

print(result)

with open(output_file, 'wb') as f:
    pickle.dump(result, f)
