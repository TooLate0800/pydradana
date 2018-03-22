#!/usr/bin/env python3

import argparse
import glob
import os
import pickle

import numpy

from pydradana import sim_analyzer, sim_configs

parser = argparse.ArgumentParser(description='calculate simulation acceptances')
parser.add_argument('path', nargs=1, help='path to yield results')
parser.add_argument('-e', default=2.2, type=float, help='beam energy', dest='e')
# 2.2 gev: 735.34935, 1.1 gev: 177.34404
parser.add_argument('-l', default=735.34935, type=float, help='luminosity per file', dest='lumi_per_file')
parser.add_argument('-o', default='form_factors.pkl', type=str, help='output file', dest='output_file')
parser.add_argument('-r', default='rad_correction.pkl', type=str, help='radiative correction file', dest='rad_cor_file')

args = vars(parser.parse_args())
path = args['path'][0]
ei = args['e']
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

lumi = n_files * lumi_per_file

with open(rad_correction_file, 'rb') as f:
    rad_correction_dict = pickle.load(f)
rad_cor = rad_correction_dict['rad_correction']
drad_cor = rad_correction_dict['error_of_rad_correction']

q2, dq2, gc, dgc = sim_analyzer.calculate_gc(ei, lumi, yield_, dyield_, rad_cor, drad_cor)

result['Q2'] = q2
result['dQ2'] = dq2
result['GC'] = gc
result['dGC'] = dgc

with open(output_file, 'wb') as f:
    pickle.dump(result, f)
