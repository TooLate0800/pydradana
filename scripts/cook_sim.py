#!/usr/bin/env python3

import argparse
import os
import pickle

from pydradana import sim_counter

parser = argparse.ArgumentParser(description='cook simulation results')
parser.add_argument('file', nargs=1, help='simulation result rootfile')
parser.add_argument('-e', default=500000, type=int, help='event amount', dest='n')
parser.add_argument('-z', default='z_correction.pkl', type=str, help='z correction file', dest='z_cor_file')

args = vars(parser.parse_args())
filename = args['file'][0]
n = args['n']
z_correction_file = args['z_cor_file']

if os.path.exists(os.path.basename(filename) + '.pkl'):
    os.remove(os.path.basename(filename) + '.pkl')

with open(z_correction_file, 'rb') as f:
    z_correction_dict = pickle.load(f)

result = sim_counter.count_yield(filename, 0, n, z_correction_dict['z_correction'])

with open(os.path.basename(filename) + '.pkl', 'wb') as f:
    pickle.dump(result, f)
