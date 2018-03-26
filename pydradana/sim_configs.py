#!/usr/bin/env python3

import numpy
from scipy import constants

# constants
m_e = constants.value('electron mass energy equivalent in MeV') * 1e-3
m_p = constants.value('proton mass energy equivalent in MeV') * 1e-3
m_d = constants.value('deuteron mass energy equivalent in MeV') * 1e-3
alpha = constants.alpha
mu_p = constants.value('proton mag. mom. to nuclear magneton ratio')
inv_fm_to_gev = constants.hbar * constants.c / constants.e * 1e6  # fm^{-1} to GeV
inv_gev_to_fm = inv_fm_to_gev
gev_to_inv_fm = 1 / inv_fm_to_gev
fm_to_inv_gev = 1 / inv_gev_to_fm
inv_gev_to_mkb = inv_gev_to_fm**2 * 1e4  # GeV^{-2} to microbarn

# PDG ID
electron = 11
photon = 22
proton = 2212
deuteron = 1000010020

# simulation configs
target_center = -3000 + 89  # mm

gem_resolution = 7.0e-2  # 70 um

error_of_acceptance = 0.003  # 0.2%
error_of_detector = 0.002 # 0.2%
error_of_event_selection = 0.002  # 0.2%
error_of_radiative_correction = 0.002  # 0.2%

# binnning
binning = {'bins': 60, 'range': (0.5, 6.5)}

def create_bins():
    bins = binning['bins']
    low, high = binning['range']
    bin_centers = numpy.linspace(low + (high - low) / bins / 2, high - (high - low) / bins / 2, bins)
    bin_edges = numpy.linspace(low, high, bins + 1)

    return bin_centers, bin_edges
