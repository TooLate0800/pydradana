#!/usr/bin/env python3

import argparse
import pickle

import numpy
import matplotlib.pyplot as plt

from pydradana import RFitter, form_factors

parser = argparse.ArgumentParser(description='calculate simulation acceptances')
parser.add_argument('file', nargs='+', help='path to form factor result file')

args = vars(parser.parse_args())
filenames = args['file']

if len(filenames) != 2:
    quit()

result = []
for filename in filenames:
    with open(filename, 'rb') as f:
        result.append(pickle.load(f))

# data selection:
# 3:-5 is 0.8 ~ 6.0 deg for 1.1 GeV (start from 0.8 due to the recoil detector)
# 2:-5 is 0.7 ~ 6.0 deg for 2.2 GeV
q2, dq2, gc, dgc = [[result[0][x][3:-5], result[1][x][2:-5]] for x in ['Q2', 'dQ2', 'GC', 'dGC']]
q2_fit, dq2_fit, gc_fit, dgc_fit = [numpy.hstack(x) for x in [q2, dq2, gc, dgc]]

index_q2 = numpy.argsort(q2_fit)

with open('bin_errors.dat', 'w') as f:
    f.write('106\n')
    for q2_i, dq2_i, gc_i, dgc_i in zip(q2_fit[index_q2], dq2_fit[index_q2], gc_fit[index_q2], dgc_fit[index_q2]):
        print(f'{q2_i:8.6f} {gc_i:12.6e} {dgc_i:12.6e}')
        f.write(f'{q2_i:8.6f} {gc_i:12.6e} {dgc_i:12.6e}\n')

fitter = RFitter()
fitter.load_data(q2=q2_fit, ge=gc_fit, dge=dgc_fit)
fitter.set_range(0, 1.5)
r, _ = fitter.fit(model=('ratio', 1, 1), method='least_squares', r0=2.094)

print(r)

gc_0, gm_0, gq_0 = form_factors.abbott_2000_1(q2_fit[index_q2])

fig = plt.Figure()
ax = plt.gca()
ax.minorticks_on()
plt.grid(True)
plt.xlabel(r'$q2 / fm^{-2}$')
plt.ylabel(r'$G_C$')
plt.errorbar(q2_fit[:52], gc_fit[:52], yerr=dgc_fit[:52], fmt='r.', label='1.1 GeV')
plt.errorbar(q2_fit[52:], gc_fit[52:], yerr=dgc_fit[52:], fmt='b.', label='2.2 GeV')
plt.plot(q2_fit[index_q2], gc_0, 'k--')
plt.legend()
plt.show()
