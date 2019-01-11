# Author: Chao Gu, 2018

# For python 2-3 compatibility
from __future__ import division, print_function
from functools import reduce

import math

import iminuit
import numpy
from numpy.random import normal, uniform
from lmfit import Minimizer, Parameters
from scipy import constants
from scipy.interpolate import interp1d

from . import form_factors

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
        s_tc_m_q2 = numpy.sqrt(self._tc)  # ignore 0
        self.q2 = (s_tc_p_q2 - s_tc_m_q2) / (s_tc_p_q2 + s_tc_m_q2)

    @staticmethod
    def _monopole(q2, p, *args, **kwargs):
        return p[0] / (1 + q2 * p[1]**2 / 6)

    @staticmethod
    def _dipole(q2, p, *args, **kwargs):
        return p[0] / (1 + q2 * p[1]**2 / 12)**2

    @staticmethod
    def _gaussian(q2, p, *args, **kwargs):
        return p[0] * numpy.exp(-q2 * p[1]**2 / 6)

    @staticmethod
    def _polynominal(q2, p, order, *args, **kwargs):
        N = order[0] + 1
        return p[0] * (1 - q2 * p[1]**2 / 6 + sum([p[i] * q2**i for i in range(2, N)]))

    @staticmethod
    def _rational(q2, p, order, *args, **kwargs):
        N = order[0] + 1
        M = order[1] + 1
        p_a = [0] + p[1:N]
        p_b = [0] + p[N:M + N]
        # notice: additional p^b_1 * Q2 in numerator
        numerator = 1 - q2 * p[1]**2 / 6 + sum([p_a[i] * q2**i for i in range(2, N)]) + p_b[1] * q2
        denominator = 1 + sum([p_b[j] * q2**j for j in range(1, M)])
        return p[0] * numerator / denominator

    @staticmethod
    def _continued_fractional(q2, p, order, *args, **kwargs):
        result = reduce(lambda x, y: 1 + y / x, [1] + [p[i] * q2 for i in range(order[0], 1, -1)])
        return p[0] / (1 + q2 * p[1] * p[1] / 6 / result)

    # Form factors from file
    @staticmethod
    def _ff_file(q2, filename, unit='GeV'):
        q2_model, ge_model = numpy.loadtxt(filename, usecols=(0, 1), unpack=True)
        if unit == 'GeV':
            q2_model = q2_model * _gev_to_inv_fm**2
        spl = interp1d(q2_model, ge_model, kind='cubic')

        return spl(q2)

    def _get_residual_func(self, model_func, n_pars, *args, **kwargs):

        def residual(pars):
            parvals = pars.valuesdict()
            result = model_func(self.q2, [parvals[f'p{i}'] for i in range(n_pars)], *args, **kwargs)
            return (result - self.ge) / self.dge

        return residual

    def _get_residual_func_minuit(self, model_func, n_pars, *args, **kwargs):

        def residual(*pars):
            result = model_func(self.q2, list(pars), *args, **kwargs)
            return numpy.sum(((result - self.ge) / self.dge)**2)

        return residual

    # Public methods
    def load_data(self, filename='bin_errors.dat', *, q2=None, ge=None, dge=None):
        if any(x is None for x in [q2, ge, dge]):
            self.q2_raw, self.ge_raw, self.dge_raw = numpy.loadtxt(filename, usecols=(0, 1, 2), skiprows=1, unpack=True)
        else:
            self.q2_raw, self.ge_raw, self.dge_raw = q2, ge, dge
        self.dq2_raw = numpy.zeros(self.q2_raw.shape)

    def print_data(self, select='raw'):
        if select == 'raw':
            for q2, ge, dge in zip(self.q2_raw, self.ge_raw, self.dge_raw):
                print(f'{q2: .10f} {ge: .10f} {dge: .10f}')
        else:
            if self.range is not None:
                self._select_q2()
                for q2, ge, dge in zip(self.q2, self.ge, self.dge):
                    print(f'{q2: .10f} {ge: .10f} {dge: .10f}')

    def set_range(self, lo=0.0, hi=0.5):
        self.range = [lo, hi]

    def gen_model(self, model='dipole', r0=2.130, *args, **kwargs):
        model_func = {
            'monopole': form_factors.monopole,
            'dipole': form_factors.dipole,
            'gaussian': form_factors.gaussian,
            'Arrington-2004': form_factors.arrington_2004,
            'Kelly-2004': form_factors.kelly_2004,
            'Venkat-2011': form_factors.venkat_2011,
            'Abbott-2000-1': form_factors.abbott_2000_1,
            'Abbott-2000-2': form_factors.abbott_2000_2,
        }.get(model, None)

        if self.q2_raw is None:
            self.load_data('bin_errors.dat')

        if model_func is not None:
            self.ge_raw, _, _ = model_func(self.q2_raw, r0)
        else:
            self.ge_raw = self._ff_file(self.q2_raw, filename=model + '.dat', *args, **kwargs)

    def add_noise(self, model='gaussian', *args, **kwargs):
        if model == 'uniform':
            self.ge_raw = self.ge_raw + uniform(-numpy.sqrt(3) * self.dge_raw, numpy.sqrt(3) * self.dge_raw)
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

        n_pars, r_guess = 2, r0
        if model == 'monopole':
            model_func = self._monopole
        elif model == 'dipole':
            model_func = self._dipole
        elif model == 'gaussian':
            model_func = self._gaussian
        elif model == 'poly':
            model_func = self._polynominal
            n_pars = order[0] + 1
        elif model == 'ratio':
            model_func = self._rational
            n_pars = order[0] + order[1] + 1
        elif model == 'cf':
            model_func = self._continued_fractional
            n_pars = order[0] + 1
        elif model == 'poly-z':
            model_func = self._polynominal
            n_pars = order[0] + 1
            r_guess = r_guess * numpy.sqrt(4 * self._tc)
        else:
            print(f'model {model} is not valid')
            return None, None, None, None

        if self.range is not None:
            self._select_q2()
        if model == 'poly-z':
            self._convert_q2_z()

        abs_r_guess = numpy.fabs(r_guess)

        if 'method' in kwargs and kwargs['method'] == 'minuit':
            residule_func = self._get_residual_func_minuit(model_func, n_pars, order)

            parameters = ['p0', 'p1']
            init_values = {}
            init_values['p0'] = 1
            init_values['error_p0'] = 0.001
            init_values['limit_p0'] = (0.95, 1.05)
            if not float_norm:
                init_values['fix_p0'] = True
            init_values['p1'] = r_guess
            init_values['error_p1'] = abs_r_guess * 0.01
            init_values['limit_p1'] = (r_guess - 0.5 * abs_r_guess, r_guess + 0.5 * abs_r_guess)
            for i in range(2, n_pars):
                parameters.append(f'p{i}')
                init_values[f'p{i}'] = 0
                init_values[f'error_p{i}'] = 0.01

            for _ in range(100):
                fitter = iminuit.Minuit(residule_func, forced_parameters=parameters, pedantic=False, print_level=0, **init_values)
                fitter.migrad()

                p1 = fitter.values['p1']
                p1_min, p1_max = init_values['limit_p1']
                if math.isclose(p1, p1_min, abs_tol=5e-2 * abs_r_guess) or math.isclose(p1, p1_max, abs_tol=5e-2 * abs_r_guess):
                    for i in range(2, n_pars):
                        init_values[f'p{i}'] = normal()
                else:
                    break

            r = fitter.values['p1']
            chisqr = fitter.fval
        else:
            residual_func = self._get_residual_func(model_func, n_pars, order)

            params = Parameters()
            params.add('p0', value=1.0, vary=float_norm, min=0.95, max=1.05)
            params.add('p1', value=r_guess, min=r_guess - 0.5 * abs_r_guess, max=r_guess + 0.5 * abs_r_guess)
            for i in range(2, n_pars):
                params.add(f'p{i}', value=0)

            fitter = Minimizer(residual_func, params)

            for _ in range(100):
                fit_result = fitter.minimize(*args, **kwargs)

                p1 = fit_result.params['p1']
                if math.isclose(p1.value, p1.min, abs_tol=5e-2 * abs_r_guess) or math.isclose(p1.value, p1.max, abs_tol=5e-2 * abs_r_guess):
                    for i in range(2, n_pars):
                        params[f'p{i}'].value = normal()
                else:
                    break

            r = p1.value
            chisqr = fit_result.chisqr

        if model == 'poly-z':
            r = r / numpy.sqrt(4 * self._tc)

        return r, chisqr
