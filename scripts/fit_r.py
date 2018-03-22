#!/usr/bin/env python3

import argparse
import glob
import os
import pickle

import numpy
import matplotlib.pyplot as plt
from scipy import constants

from pydradana import RFitter, sim_configs
# from pydradana.form_factors import abbott_2000_1

from cal_gc import cal_gc

parser = argparse.ArgumentParser(description='fit radius')
parser.add_argument('path', nargs='+', help='path to yield results')

args = vars(parser.parse_args())
paths = args['path']

if len(paths) != 2:
    quit()

# load yield results
yields_1gev = []
for filename in glob.glob(os.path.join(paths[0], '*.pkl')):
    with open(filename, 'rb') as f:
        result = pickle.load(f)
    yields_1gev.append(result['hist_theta'])

yields_2gev = []
for filename in glob.glob(os.path.join(paths[1], '*.pkl')):
    with open(filename, 'rb') as f:
        result = pickle.load(f)
    yields_2gev.append(result['hist_theta'])

# load radiative correction
with open(os.path.join('rad_correction', 'rad_cor_1gev.pkl'), 'rb') as f:
    rad_correction_dict = pickle.load(f)
rad_cor_1gev = rad_correction_dict['rad_correction']
drad_cor_1gev = rad_correction_dict['error_of_rad_correction']

with open(os.path.join('rad_correction', 'rad_cor_2gev.pkl'), 'rb') as f:
    rad_correction_dict = pickle.load(f)
rad_cor_2gev = rad_correction_dict['rad_correction']
drad_cor_2gev = rad_correction_dict['error_of_rad_correction']

n = (len(yields_1gev) + len(yields_2gev)) // 2
r_results = []

for i_fit in range(1000):
    yield_1gev = numpy.zeros(sim_configs.binning['bins'])
    yield_2gev = numpy.zeros(sim_configs.binning['bins'])
    for i in range(n):
        yield_1gev += yields_1gev[numpy.random.randint(n)]
        yield_2gev += yields_2gev[numpy.random.randint(n)]
    systematics = numpy.sqrt(sim_configs.error_of_acceptance**2 + sim_configs.error_of_detector**2 + sim_configs.error_of_event_selection**2)
    #yield_1gev = yield_1gev * numpy.random.normal(1, systematics)
    #yield_2gev = yield_2gev * numpy.random.normal(1, systematics)

    dyield_1gev = numpy.sqrt(yield_1gev)
    dyield_2gev = numpy.sqrt(yield_2gev)

    lumi_1gev = n * 177.34404
    lumi_2gev = n * 735.34935

    q2_1gev, dq2_1gev, gc_1gev, dgc_1gev = cal_gc(1.1, yield_1gev, dyield_1gev, lumi_1gev, rad_cor_1gev, drad_cor_1gev)
    q2_2gev, dq2_2gev, gc_2gev, dgc_2gev = cal_gc(2.2, yield_2gev, dyield_2gev, lumi_2gev, rad_cor_2gev, drad_cor_2gev)

    # data selection:
    # 2:-5 is 0.7 ~ 6.0 deg
    q2_1gev, dq2_1gev, gc_1gev, dgc_1gev = [x[3:-5] for x in [q2_1gev, dq2_1gev, gc_1gev, dgc_1gev]]
    q2_2gev, dq2_2gev, gc_2gev, dgc_2gev = [x[2:-5] for x in [q2_2gev, dq2_2gev, gc_2gev, dgc_2gev]]
    q2_fit, dq2_fit, gc_fit, dgc_fit = [
        numpy.hstack(x) for x in [[q2_1gev, q2_2gev], [dq2_1gev, dq2_2gev], [gc_1gev, gc_2gev], [dgc_1gev, dgc_2gev]]
    ]

    # index_q2 = numpy.argsort(q2_fit)

    # with open('bin_errors.dat', 'w') as f:
    #     f.write('105\n')
    #     for q2_i, dq2_i, gc_i, dgc_i in zip(q2_fit[index_q2], dq2_fit[index_q2], gc_fit[index_q2], dgc_fit[index_q2]):
    #         print('{:8.6f} {:12.6e} {:12.6e}'.format(q2_i, gc_i, dgc_i))
    #         f.write('{:8.6f} {:12.6e} {:12.6e}\n'.format(q2_i, gc_i, dgc_i))

    fitter = RFitter()
    fitter.load_data(q2=q2_fit, ge=gc_fit, dge=dgc_fit)
    fitter.set_range(0.0, 1.5)
    r, _ = fitter.fit(model=('ratio', 1, 1), method='least_squares', r0=2.130)

    print(i_fit, r)

    r_results.append(r)

result = numpy.array(r_results)

r_ave = numpy.average(result)
r_error = numpy.sqrt(numpy.sum((result - r_ave)**2) / (len(result) - 1))

font = {'size': 12}

fig = plt.Figure()
ax = plt.gca()
ax.minorticks_on()
#plt.xlabel(r'$q2 / fm^{-2}$')
#plt.xscale('log')
#plt.errorbar(q2_1, gc_1, yerr=dgc_1, fmt='r.')
#plt.errorbar(q2_2, gc_2, yerr=dgc_2, fmt='b.')
#plt.plot(q2_fit[index_q2], gc_0, 'k--')
plt.hist(result, bins=50, range=(2.03, 2.13), histtype='step')
plt.text(0.1, 0.8, r'$r\,={:6.4f}$'.format(r_ave) + '\n' + r'$\sigma={:6.4f}$'.format(r_error), transform=ax.transAxes, fontdict=font)
plt.show()
