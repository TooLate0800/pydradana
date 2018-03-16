#!/usr/bin/env python3

import argparse
import os
import pickle

from pydradana import sim_counter

parser = argparse.ArgumentParser(description='calculate simulation yield')
parser.add_argument('file', nargs=1, help='rootfile')
parser.add_argument('-e', default=500000, type=int, help='event amount', dest='n')
parser.add_argument('-a', default='acceptance.pkl', type=str, help='acceptance file', dest='acc_file')

args = vars(parser.parse_args())
filename = args['file'][0]
n = args['n']
acceptance_file = args['acc_file']

if os.path.exists(os.path.basename(filename) + '.pkl'):
    os.remove(os.path.basename(filename) + '.pkl')

with open(acceptance_file, 'rb') as f:
    acceptance = pickle.load(f)

result = sim_counter.get_yield(filename, 0, n, acceptance['z_correction'])

with open(os.path.basename(filename) + '.pkl', 'wb') as f:
    pickle.dump(result, f)
