#!/usr/bin/env python3

import argparse
import glob
import os
import pickle

import numpy

from pydradana import sim_configs, sim_counter

parser = argparse.ArgumentParser(description='calculate simulation acceptances')
parser.add_argument('path', nargs=1, help='path to rootfiles or results')
parser.add_argument('-e', default=500000, type=int, help='event amount', dest='n')
parser.add_argument('-c', action='store_true', help='combine results', dest='combine')

args = vars(parser.parse_args())
path = args['path'][0]
n = args['n']
combine = args['combine']

if not combine:
    result = sim_counter.get_acceptance(path, 0, n)
    with open(os.path.basename(path) + '.pkl', 'wb') as f:
        pickle.dump(result, f)
else:
    hist_theta = numpy.zeros(sim_configs.binning['bins'])
    hist_theta_good = numpy.zeros(sim_configs.binning['bins'])
    hist_z_good = numpy.zeros(sim_configs.binning['bins'])
    for filename in glob.glob(os.path.join(path, '*.pkl')):
        with open(filename, 'rb') as f:
            result = pickle.load(f)
        hist_theta += result['hist_theta']
        hist_theta_good += result['hist_theta_good']
        hist_z_good += result['hist_z_good']

    acceptance = hist_theta_good / hist_theta
    error_of_acceptance = numpy.sqrt(1 / hist_theta + 1 / hist_theta_good) * acceptance
    averaged_z_center = hist_z_good / hist_theta_good

    bins = sim_configs.binning['bins']
    low, high = sim_configs.binning['range']
    bin_centers = numpy.linspace(low + (high - low) / bins / 2, high - (high - low) / bins / 2, bins)
    z_correction = numpy.polyfit(bin_centers * numpy.pi / 180.0, averaged_z_center, 1)

    result = {}
    result['acceptance'] = acceptance
    result['error_of_acceptance'] = error_of_acceptance
    result['z_correction'] = z_correction

    with open(os.path.join(path, 'acceptance.pkl'), 'wb') as f:
        pickle.dump(result, f)
