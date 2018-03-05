#!/usr/bin/env python

# For python 2-3 compatibility
from __future__ import division, print_function
from functools import reduce

import math
import numpy as np
from numpy.random import normal, uniform
from lmfit import Minimizer, Parameters, fit_report
from scipy.interpolate import interp1d

# Constants
_mp = 0.938272    # GeV
_md = 1.875612928    # GeV
_fm = 0.1973269718    # fm^{-1} to GeV

class RFitter():
    def __init__(self):
        self._Tc = 4 * 0.14**2 / _fm**2    # 4 * m_pi**2
        self.Q2_raw, self.sQ2_raw, self.GE_raw, self.dGE_raw = None, None, None, None
        self.Q2, self.sQ2, self.GE, self.dGE = None, None, None, None
        self.range = None

    # Private methods
    def _select_Q2(self):
        select = (self.Q2_raw > self.range[0]) & (self.Q2_raw < self.range[1])
        self.Q2 = self.Q2_raw[select]
        self.sQ2 = self.sQ2_raw[select]
        self.GE = self.GE_raw[select]
        self.dGE = self.dGE_raw[select]

    def _convert_Q2_z(self):
        s_Tc_p_Q2 = np.sqrt(self._Tc + self.Q2)
        s_Tc_m_Q2 = math.sqrt(self._Tc)    # ignore 0
        self.Q2 = (s_Tc_p_Q2 - s_Tc_m_Q2) / (s_Tc_p_Q2 + s_Tc_m_Q2)

    @staticmethod
    def _monopole(q2, p, *args, **kwargs):
        return p[0] / (1 + q2 / p[1])
    
    @staticmethod
    def _dipole(q2, p, *args, **kwargs):
        return p[0] / (1 + q2 / p[1])**2

    @staticmethod
    def _gaussian(q2, p, *args, **kwargs):
        return p[0] * np.exp(-q2 / p[1])

    @staticmethod
    def _polynominal(q2, p, order, *args, **kwargs):
        N = order[0] + 1
        return p[0] * (1 + sum([p[i] * q2**i for i in range(1, N)]))

    @staticmethod
    def _rational(q2, p, order, *args, **kwargs):
        N = order[0] + 1
        M = order[1] + 1
        p_a = [0] + p[1:N]
        p_b = [0] + p[N:M+N]
        numerator = 1 + sum([p_a[i] * q2**i for i in range(1, N)]) + p_b[1] * q2    # notice: additional p^b_1 * Q2 in numerator
        denominator = 1 + sum([p_b[j] * q2**j for j in range(1, M)])
        return p[0] * numerator / denominator   

    @staticmethod
    def _continued_fractional(q2, p, order, *args, **kwargs):
        return p[0] / reduce(lambda x, y: 1 + y / x, [1] + [p[i] * q2 for i in range(order[0], 0 , -1)])

    # Proton form factors
    @staticmethod
    def _arrington_2004(q2, *args, **kwargs):
        # J. Arrington, Phys. Rev. C 69(2004)022201
        q2_gev = q2 * _fm**2
        a = [0, 3.226, 1.508, -0.3773, 0.611, -0.1853, 1.596e-2]
        return 1 / (1 + sum([a[i] * q2_gev**i for i in range(1, 7)]))

    @staticmethod
    def _kelly_2004(q2, *args, **kwargs):
        # J. J. Kelly, Phys. Rev. C 70(2004)068202
        tau = q2 * _fm**2 / (4 * _mp**2)
        a = [0, -0.24]
        b = [0, 10.98, 12.82, 21.97]
        return (1 + tau * a[1]) / (1 + sum([b[i] * tau**i for i in range(1, 4)]))

    @staticmethod
    def _venkat_2011(q2, *args, **kwargs):
        # S. Venkat, J. Arrington, G. A. Miller, and X. Zhan, Phys. Rev. C 83(2011)015203
        tau = q2 * _fm**2 / (4 * _mp**2)
        a = [0, 2.90966, -1.11542229, 3.866171e-2]
        b = [0, 14.5187212, 40.88333, 99.999998, 4.579e-5, 10.3580447]
        return (1 + sum([a[i] * tau**i for i in range(1, 4)])) / (1 + sum([b[i] * tau**i for i in range(1, 6)]))
    
    # Deuteron form factors
    @staticmethod
    def _abbott_2000_1(q2, *args, **kwargs):
        # Parameterization I in Eur. Phys. J A 7(2000)421
        GC0, QC0 = 1, 4.21
        aC = [0, 6.740e-1,  2.246e-2,  9.806e-3, -2.709e-4,  3.793e-6]
        return GC0 * (1 - q2 / QC0**2) / (1 + sum([aC[i] * q2**i for i in range(1, 6)]))

    @staticmethod
    def _abbott_2000_2(q2, *args, **kwargs):
        # Parameterization II in Eur. Phys. J A 7(2000)421
        eta = q2 / (4 * (_md / _fm)**2)
        delta = (0.89852 / _fm)**2
        G = 1 / (1 + q2 / (4 * delta))**2
        a = [1.57057, 12.23792, -42.04576, 27.92014]
        alpha = [1.52501, 8.75139, 15.97777, 23.20415]
        b = [0.07043, 0.14443, -0.27343, 0.05856]
        beta = [43.67795, 30.05435, 16.43075, 2.80716]
        c = [-0.16577, 0.27557, -0.05382, -0.05598]
        gamma = [1.87055, 14.95683, 28.04312, 41.12940]
        g0 = sum([a[i] / (alpha[i] + q2) for i in range(4)])
        g1 = np.sqrt(q2) * sum([b[i] / (beta[i] + q2) for i in range(4)])
        g2 = q2 * sum([c[i] / (gamma[i] + q2) for i in range(4)])
        return G**2 / (2 * eta + 1) * ((1 - 2 / 3 * eta) * g0 + 8 / 3 * np.sqrt(2 * eta) * g1 + 2 / 3 * (2 * eta - 1) * g2)

    # Form factors from file
    @staticmethod
    def _ff_file(q2, file_name):
        q2_model, ge_model = np.loadtxt(file_name, usecols=(0, 1), unpack=True)
        q2_model = q2_model / _fm**2
        spl = interp1d(q2_model, ge_model, kind='cubic')
        return spl(q2)
        
    def _get_residual_func(self, model_func, n_pars, *args, **kwargs):
        def residual(pars):
            parvals = pars.valuesdict()
            result = model_func(self.Q2, [parvals['p{}'.format(i)] for i in range(n_pars)], *args, **kwargs)
            return (result - self.GE) / self.dGE
        return residual

    # Public methods
    def load_data(self, file_name='bin_errors.dat'):
        self.Q2_raw, self.GE_raw, self.dGE_raw = np.loadtxt(file_name, usecols=(0, 1, 2), skiprows=1, unpack=True)
        self.Q2_raw = self.Q2_raw
        self.sQ2_raw = np.zeros(self.Q2_raw.shape)

    def print_data(self, select='raw'):
        if select == 'raw':
            for i in range(len(self.Q2_raw)):
                print('{: .10f} {: .10f} {: .10f}'.format(self.Q2_raw[i], self.GE_raw[i], self.dGE_raw[i]))
        else:
            if self.range is not None:
                self._select_Q2()
                for i in range(len(self.Q2)):
                    print('{: .10f} {: .10f} {: .10f}'.format(self.Q2[i], self.GE[i], self.dGE[i]))

    def set_range(self, lo=0.0, hi=0.5):
        self.range = [lo, hi]

    def gen_model(self, model='dipole', r=2.130):
        model_func = {
            'monopole': self._monopole,
            'dipole': self._dipole,
            'gaussian': self._gaussian,
            'Arrington-2004': self._arrington_2004,
            'Kelly-2004': self._kelly_2004,
            'Venkat-2011': self._venkat_2011,
            'Abbott-2000-1': self._abbott_2000_1,
            'Abbott-2000-2': self._abbott_2000_2
        }.get(model, None)

        p = [1, 0]
        if model == 'monopole' or model == 'gaussian':
            p[1] = 6 / r**2
        elif model == 'dipole':
            p[1] = 12 / r**2
        
        if self.Q2_raw is None: self.load_data('bin_errors.dat')

        if model_func is not None:
            self.GE_raw = model_func(self.Q2_raw, p)
        else:
            self.GE_raw = self._ff_file(self.Q2_raw, file_name=model+'.dat')

    def add_noise(self, model='gaussian', *args, **kwargs):
        if model == 'uniform':
            self.GE_raw = self.GE_raw + uniform(-math.sqrt(3) * self.dGE_raw, math.sqrt(3) * self.dGE_raw)
        elif model == 'gaussian':
            self.GE_raw = self.GE_raw + normal(0, self.dGE_raw)
        elif model == 'scale':
            noise = normal(1, args[0])
            self.GE_raw = self.GE_raw * noise
        elif model == 'shift':
            self.GE_raw[:int(args[0])] = self.GE_raw[:int(args[0])] + args[1] * self.dGE_raw[:int(args[0])]
            self.GE_raw[int(args[0]):] = self.GE_raw[int(args[0]):] - args[1] * self.dGE_raw[int(args[0]):]

    def fit(self, model='dipole', float_norm=True, r_guess=2.130, *args, **kwargs):
        order = model[1:] if isinstance(model, (tuple, list)) else None
        model = model[0] if isinstance(model, (tuple, list)) else model

        n_pars, p1_guess = 2, 6 / r_guess**2
        if model == 'monopole':
            model_func = self._monopole
        elif model == 'dipole':
            model_func = self._dipole
            p1_guess = 12 / r_guess**2
        elif model == 'gaussian':
            model_func = self._gaussian
        elif model == 'poly':
            model_func = self._polynominal
            n_pars = order[0] + 1
            p1_guess = -r_guess**2 / 6
        elif model == 'ratio':
            model_func = self._rational
            n_pars = order[0] + order[1] + 1
            p1_guess = -r_guess**2 / 6
        elif model == 'cf':
            model_func = self._continued_fractional
            n_pars = order[0] + 1
            p1_guess = r_guess**2 / 6
        elif model == 'poly-z':
            model_func = self._polynominal
            n_pars = order[0] + 1
            p1_guess = -r_guess**2 * self._Tc / 1.5
        else:
            print('model {} is not valid'.format(model))
            return None, None, None, None

        if self.range is not None: self._select_Q2()   
        if model == 'poly-z': self._convert_Q2_z()

        residual_func = self._get_residual_func(model_func, n_pars, order)

        params = Parameters()
        params.add('p0', value=1.0, vary=float_norm, min=0.95, max=1.05)
        params.add('p1', value=p1_guess, min=p1_guess-0.5*math.fabs(p1_guess), max=p1_guess+0.5*math.fabs(p1_guess))
        for i in range(2, n_pars):
            params.add('p{}'.format(i), value=0)

        fitter = Minimizer(residual_func, params)

        def is_close(a, a0, tolerance=1e-4):
            return math.fabs(a - a0) < tolerance * math.fabs(a0)
          
        for i in range(5):
            fit_result = fitter.minimize(*args, **kwargs)
            
            par1 = fit_result.params['p1']
            if is_close(par1.value, par1.min, 1e-2) or is_close(par1.value,par1.max, 1e-2):
                for i in range(2, n_pars):
                    params['p{}'.format(i)].value = normal()
            else:
                if __name__ == '__main__':    # debug info
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
            r = math.sqrt(-1.5 * p1 / self._Tc)

        return r, chisqr

if __name__ == '__main__':
    f = RFitter()
    f.load_data('bin_errors.dat')
    f.gen_model('dipole')
    f.add_noise('gaussian')
    f.set_range(0, 2)
    r, chisqr = f.fit(model='dipole', method='least_squares')
    print('r = {:10.6f}, chisqr = {:10.6f}'.format(r, chisqr))
    