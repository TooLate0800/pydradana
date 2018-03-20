#!/usr/bin/env python3

import argparse
import os
import pickle

from pydradana import sim_counter

parser = argparse.ArgumentParser(description='cook acceptance simulation results')
parser.add_argument('file', nargs=1, help='acceptance simulation result rootfile')
parser.add_argument('-e', default=500000, type=int, help='event amount', dest='n')

args = vars(parser.parse_args())
filename = args['file'][0]
n = args['n']

result_acc = sim_counter.count_good_events(filename, 0, n)
result_z = sim_counter.count_z_ave(filename, 0, n)
result = result_acc.copy()
result.update(result_z)

print(result)

with open(os.path.basename(filename) + '.pkl', 'wb') as f:
    pickle.dump(result, f)
