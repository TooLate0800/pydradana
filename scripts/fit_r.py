#!/usr/bin/env python3

import argparse
import pickle

import numpy
import matplotlib.pyplot as plt
from scipy import constants

from pydradana import RFitter
from pydradana.form_factors import abbott_2000_1

_m_d = constants.value('deuteron mass energy equivalent in MeV') * 1e-3
_m2_d = _m_d**2
_inv_fm_to_gev = constants.hbar * constants.c / constants.e * 1e6  # fm^{-1} to GeV
_gev_to_inv_fm = 1 / _inv_fm_to_gev

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
# 5:-5 is 1.0 ~ 6.0 deg for 1.1 GeV (start from 1.0 due to the recoil detector behavior)
# 2:-5 is 0.7 ~ 6.0 deg for 2.2 GeV
q2, dq2, gc, dgc = [[result[0][x][5:-5], result[1][x][2:-5]] for x in ['Q2', 'dQ2', 'GC', 'dGC']]
q2_fit, dq2_fit, gc_fit, dgc_fit = [numpy.hstack(x) for x in [q2, dq2, gc, dgc]]

index_q2 = numpy.argsort(q2_fit)

with open('bin_errors.dat', 'w') as f:
    f.write('103\n')
    for q2_i, dq2_i, gc_i, dgc_i in zip(q2_fit[index_q2], dq2_fit[index_q2], gc_fit[index_q2], dgc_fit[index_q2]):
        print('{:8.6f} {:12.6e} {:12.6e}'.format(q2_i, gc_i, dgc_i))
        f.write('{:8.6f} {:12.6e} {:12.6e}\n'.format(q2_i, gc_i, dgc_i))

fitter = RFitter()
fitter.load_data(q2=q2_fit, ge=gc_fit, dge=dgc_fit)
fitter.set_range(0.03, 1.5)
r, _ = fitter.fit(model=('ratio', 1, 1), method='least_squares', r0=2.094)

print(r)

gc_0, gm_0, gq_0 = abbott_2000_1(q2_fit[index_q2])

fig = plt.Figure()
ax = plt.gca()
ax.minorticks_on()
plt.xlabel(r'$q2 / fm^{-2}$')
plt.xscale('log')
plt.errorbar(q2_fit, gc_fit, yerr=dgc_fit, fmt='b.')
plt.plot(q2_fit[index_q2], gc_0, 'r--')
plt.show()
