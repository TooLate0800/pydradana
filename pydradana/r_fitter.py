#!/usr/bin/env python3

# For python 2-3 compatibility
from __future__ import division, print_function
from functools import reduce

import math

import numpy
from numpy.random import normal, uniform
from lmfit import Minimizer, Parameters, fit_report
from scipy import constants
from scipy.interpolate import interp1d

from . import _form_factors

__all__ = ['RFitter']

# Constants
_inv_fm_to_gev = constants.hbar * constants.c / constants.e * 1e6  # fm^{-1} to GeV
_gev_to_inv_fm = 1 / _inv_fm_to_gev


class RFitter(object):

    def __init__(self):
        self._tc = 4 * 0.140**2 * _gev_to_inv_fm**2  # 4 * m_pi**2
        self.q2_raw, self.dq2_raw, self.ge_raw, self.dge_raw = None, None, None, None
        self.q2, self.dq2, self.ge, self.dge = None, None, None, None
        self.range = None

    # Private methods
    def _select_q2(self):
        select = (self.q2_raw > self.range[0]) & (self.q2_raw < self.range[1])
        self.q2 = self.q2_raw[select]
        self.dq2 = self.dq2_raw[select]
        self.ge = self.ge_raw[select]
        self.dge = self.dge_raw[select]

    def _convert_q2_z(self):
        s_tc_p_q2 = numpy.sqrt(self._tc + self.q2)
        s_tc_m_q2 = math.sqrt(self._tc)  # ignore 0
        self.q2 = (s_tc_p_q2 - s_tc_m_q2) / (s_tc_p_q2 + s_tc_m_q2)

    @staticmethod
    def _monopole(q2, p, *args, **kwargs):
        return p[0] / (1 + q2 / p[1])

    @staticmethod
    def _dipole(q2, p, *args, **kwargs):
        return p[0] / (1 + q2 / p[1])**2

    @staticmethod
    def _gaussian(q2, p, *args, **kwargs):
        return p[0] * numpy.exp(-q2 / p[1])

    @staticmethod
    def _polynominal(q2, p, order, *args, **kwargs):
        N = order[0] + 1
        return p[0] * (1 + sum([p[i] * q2**i for i in range(1, N)]))

    @staticmethod
    def _rational(q2, p, order, *args, **kwargs):
        N = order[0] + 1
        M = order[1] + 1
        p_a = [0] + p[1:N]
        p_b = [0] + p[N:M + N]
        # notice: additional p^b_1 * Q2 in numerator
        numerator = 1 + sum([p_a[i] * q2**i for i in range(1, N)]) + p_b[1] * q2
        denominator = 1 + sum([p_b[j] * q2**j for j in range(1, M)])
        return p[0] * numerator / denominator

    @staticmethod
    def _continued_fractional(q2, p, order, *args, **kwargs):
        return p[0] / reduce(lambda x, y: 1 + y / x, [1] + [p[i] * q2 for i in range(order[0], 0, -1)])

    # Form factors from file
    @staticmethod
    def _ff_file(q2, file_name, unit='GeV'):
        q2_model, ge_model = numpy.loadtxt(file_name, usecols=(0, 1), unpack=True)
        if unit == 'GeV':
            q2_model = q2_model * _gev_to_inv_fm**2
        spl = interp1d(q2_model, ge_model, kind='cubic')
        return spl(q2)

    def _get_residual_func(self, model_func, n_pars, *args, **kwargs):

        def residual(pars):
            parvals = pars.valuesdict()
            result = model_func(self.q2, [parvals['p{}'.format(i)] for i in range(n_pars)], *args, **kwargs)
            return (result - self.ge) / self.dge

        return residual

    # Public methods
    def load_data(self, file_name='bin_errors.dat'):
        self.q2_raw, self.ge_raw, self.dge_raw = numpy.loadtxt(file_name, usecols=(0, 1, 2), skiprows=1, unpack=True)
        self.q2_raw = self.q2_raw
        self.dq2_raw = numpy.zeros(self.q2_raw.shape)

    def print_data(self, select='raw'):
        if select == 'raw':
            for i in range(len(self.q2_raw)):
                print('{: .10f} {: .10f} {: .10f}'.format(self.q2_raw[i], self.ge_raw[i], self.dge_raw[i]))
        else:
            if self.range is not None:
                self._select_q2()
                for i in range(len(self.q2)):
                    print('{: .10f} {: .10f} {: .10f}'.format(self.q2[i], self.ge[i], self.dge[i]))

    def set_range(self, lo=0.0, hi=0.5):
        self.range = [lo, hi]

    def gen_model(self, model='dipole', r0=2.130, *args, **kwargs):
        model_func = {
            'monopole': _form_factors.monopole,
            'dipole': _form_factors.dipole,
            'gaussian': _form_factors.gaussian,
            'Arrington-2004': _form_factors.arrington_2004,
            'Kelly-2004': _form_factors.kelly_2004,
            'Venkat-2011': _form_factors.venkat_2011,
            'Abbott-2000-1': _form_factors.abbott_2000_1,
            'Abbott-2000-2': _form_factors.abbott_2000_2,
        }.get(model, None)

        if self.q2_raw is None:
            self.load_data('bin_errors.dat')

        if model_func is not None:
            self.ge_raw, _, _ = model_func(self.q2_raw, r0)
        else:
            self.ge_raw = self._ff_file(self.q2_raw, file_name=model + '.dat', *args, **kwargs)

    def add_noise(self, model='gaussian', *args, **kwargs):
        if model == 'uniform':
            self.ge_raw = self.ge_raw + uniform(-math.sqrt(3) * self.dge_raw, math.sqrt(3) * self.dge_raw)
        elif model == 'gaussian':
            self.ge_raw = self.ge_raw + normal(0, self.dge_raw)
        elif model == 'scale':
            noise = normal(1, args[0])
            self.ge_raw = self.ge_raw * noise
        elif model == 'shift':
            self.ge_raw[:int(args[0])] = self.ge_raw[:int(args[0])] + args[1] * self.dge_raw[:int(args[0])]
            self.ge_raw[int(args[0]):] = self.ge_raw[int(args[0]):] - args[1] * self.dge_raw[int(args[0]):]

    def fit(self, model='dipole', float_norm=True, r0=2.130, *args, **kwargs):
        order = model[1:] if isinstance(model, (tuple, list)) else None
        model = model[0] if isinstance(model, (tuple, list)) else model

        n_pars, p1_guess = 2, 6 / r0**2
        if model == 'monopole':
            model_func = self._monopole
        elif model == 'dipole':
            model_func = self._dipole
            p1_guess = 12 / r0**2
        elif model == 'gaussian':
            model_func = self._gaussian
        elif model == 'poly':
            model_func = self._polynominal
            n_pars = order[0] + 1
            p1_guess = -r0**2 / 6
        elif model == 'ratio':
            model_func = self._rational
            n_pars = order[0] + order[1] + 1
            p1_guess = -r0**2 / 6
        elif model == 'cf':
            model_func = self._continued_fractional
            n_pars = order[0] + 1
            p1_guess = r0**2 / 6
        elif model == 'poly-z':
            model_func = self._polynominal
            n_pars = order[0] + 1
            p1_guess = -r0**2 * self._tc / 1.5
        else:
            print('model {} is not valid'.format(model))
            return None, None, None, None

        if self.range is not None:
            self._select_q2()
        if model == 'poly-z':
            self._convert_q2_z()

        residual_func = self._get_residual_func(model_func, n_pars, order)

        params = Parameters()
        params.add('p0', value=1.0, vary=float_norm, min=0.95, max=1.05)
        params.add('p1', value=p1_guess, min=p1_guess - 0.5 * math.fabs(p1_guess), max=p1_guess + 0.5 * math.fabs(p1_guess))
        for i in range(2, n_pars):
            params.add('p{}'.format(i), value=0)

        fitter = Minimizer(residual_func, params)

        def is_close(a, a0, tolerance=1e-4):
            return math.fabs(a - a0) < tolerance * math.fabs(a0)

        for i in range(5):
            fit_result = fitter.minimize(*args, **kwargs)

            par1 = fit_result.params['p1']
            if is_close(par1.value, par1.min, 1e-2) or is_close(par1.value, par1.max, 1e-2):
                for j in range(2, n_pars):
                    params['p{}'.format(j)].value = normal()
            else:
                if __name__ == '__main__':  # debug info
                    print(fit_report(fit_result))
                break

        p1 = par1.value
        chisqr = fit_result.chisqr
        if model == 'monopole' or model == 'gaussian':
            r = math.sqrt(6 / p1)
        elif model == 'dipole':
            r = math.sqrt(12 / p1)
        elif model == 'poly' or model == 'ratio':
            r = math.sqrt(-6 * p1)
        elif model == 'cf':
            r = math.sqrt(6 * p1)
        elif model == 'poly-z':
            r = math.sqrt(-1.5 * p1 / self._tc)

        return r, chisqr
