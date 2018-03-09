#!/usr/bin/env python

# For python 2-3 compatibility
from __future__ import division, print_function

import os
import shutil

import numpy
import uproot

__all__ = ['SimReader']

_cache_dir = '/tmp/uproot_cache'
try:
    os.rmdir(os.path.join(_cache_dir, 'order'))
    raise FileNotFoundError
except FileNotFoundError:
    if os.path.exists(_cache_dir):
        shutil.rmtree(_cache_dir)
    _disk_cache = uproot.cache.DiskCache.create(50 * 1024**3, _cache_dir)  # 50 GB disk cache
except OSError as e:
    if e.errno == 66:
        _disk_cache = uproot.cache.DiskCache.join(_cache_dir)
    else:
        raise
_cache = uproot.cache.MemoryCache(8 * 1024**3, spillover=_disk_cache, spill_immediately=True)


class _JaggedArray():

    def __init__(self, jarray):
        self._jarray = jarray
        self._data = {}

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise KeyError(key)
        jarray_at_key = [
            self._jarray.content[self._jarray.starts[i] + key] if self._jarray.starts[i] + key < self._jarray.stops[i] else numpy.nan
            for i in range(len(self._jarray))
        ]
        self._data[key] = numpy.array(jarray_at_key)
        return self._data[key]


class _Detector():

    def __init__(self, tree, name, var_dict):
        self._tree = tree
        self._name = name
        self._var_dict = var_dict

    def __getattr__(self, name):
        if name in self._var_dict:
            var_data = self._tree.array(self._var_dict[name], cache=_cache)
            if isinstance(var_data, uproot.interp.jagged.JaggedArray):
                data = _JaggedArray(var_data)
            else:
                data = var_data
            setattr(self, name, data)
            return data
        raise AttributeError(name)


class SimReader():

    def __init__(self, file_name):
        self.open(file_name)

    def _add_attrs(self):
        for det, var_dict in self._structure.items():
            option = _Detector(self._tree, det, var_dict)
            setattr(self, det, option)

    @property
    def detectors(self):
        return list(self._structure.keys())

    def open(self, file_name):
        self._tree = uproot.open(file_name)['T']
        # convert the first layer tree structure to a dictionary
        self._structure = {}
        for key in self._tree.keys():
            key_decode = key.decode('ascii').split('.')
            if len(key_decode) == 2:
                det, var = key_decode
            else:
                continue
            if det not in self._structure:
                self._structure[det] = {}
            self._structure[det][var] = (det + '.' + var).encode('ascii')
        self._add_attrs()


if __name__ == '__main__':
    r = SimReader('sim.root')
    print(r.GUN.N)
