#!/usr/bin/env python3

electron = 11
photon = 22
proton = 2212
deuteron = 1000010020

target_center = -3000 + 89  # mm

gem_resolution = 7.0e-2  # 70 um

error_of_acceptance = 0.003  # 0.2%
error_of_detector = 0.002 # 0.2%
error_of_event_selection = 0.002  # 0.2%
error_of_radiative_correction = 0.002  # 0.2%

binning = {'bins': 60, 'range': (0.5, 6.5)}
