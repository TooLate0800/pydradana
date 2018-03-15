#!/usr/bin/env python3

import argparse
import os
import pickle

from pydradana import sim_counter

parser = argparse.ArgumentParser(description='calculate simulation yield')
parser.add_argument('path', nargs=1, help='path to rootfiles')
parser.add_argument('-e', default=500000, type=int, help='event amount', dest='n')
parser.add_argument('-a', default='acceptance.pkl', type=str, help='acceptance file', dest='acc_file')

args = vars(parser.parse_args())
path = args['path'][0]
n = args['n']
acc_file = args['acc_file']

with open('acc_file', 'rb') as f:
    acceptance = pickle.load(f)

result = sim_counter.get_yield(path, 0, n, acceptance['z_correction'])
with open(os.path.basename(path) + '.pkl', 'wb') as f:
    pickle.dump(result, f)
