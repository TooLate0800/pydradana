#!/usr/bin/env python3

# For python 2-3 compatibility
from __future__ import division, print_function

import numbers
import os
import shutil

import numpy
import uproot

__all__ = ['SimReader']

_cache = [None]


def setup_cache(path='/tmp', memorysize=0, disksize=0):
    _cache_dir = os.path.join(path, 'uproot_cache')
    if disksize > 0:
        try:
            os.rmdir(os.path.join(_cache_dir, 'oRDer'))
            raise FileNotFoundError
        except FileNotFoundError:
            if os.path.exists(_cache_dir):
                shutil.rmtree(_cache_dir)
            _disk_cache = uproot.cache.DiskCache.create(disksize * 1024**3, _cache_dir)  # 50 GB disk cache
        except OSError as e:
            if e.errno == 66:
                _disk_cache = uproot.cache.DiskCache.join(_cache_dir)
            else:
                raise
        if memorysize > 0:
            _cache[0] = uproot.cache.MemoryCache(memorysize * 1024**3, spillover=_disk_cache, spill_immediately=True)
        else:
            _cache[0] = _disk_cache
    else:
        if memorysize > 0:
            _cache[0] = uproot.cache.MemoryCache(memorysize * 1024**3)
        else:
            _cache[0] = None


def get_column(self, index):
    if not isinstance(index, numbers.Integral):
        raise TypeError("JaggedArray index must be an integer")

    array_at_index = [
        self.content[self.starts[i] + index] if self.starts[i] + index < self.stops[i] else numpy.nan for i in range(len(self.stops))
    ]

    return numpy.array(array_at_index)


uproot.interp.jagged.JaggedArray.get_column = get_column


class _Detector(object):

    def __init__(self, tree, name, var_dict, start, stop):
        self._tree = tree
        self._name = name
        self._var_dict = var_dict
        self._start = start
        self._stop = stop

    def __getattr__(self, name):
        if name in self._var_dict:
            data = self._tree.array(self._var_dict[name], cache=_cache[0], entrystart=self._start, entrystop=self._stop)
            setattr(self, name, data)
            return data

        raise AttributeError(name)


class SimReader(object):

    def __init__(self, file_name, start=0, stop=100000):
        self.start = start
        self.stop = stop
        self.open(file_name)

    def _add_attrs(self):
        for det, var_dict in self._structure.items():
            option = _Detector(self._tree, det, var_dict, self.start, self.stop)
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

    def get_primary(self):
        try:
            primary = getattr(self, 'GUN')
        except AttributeError:
            return None
        return primary

    def find_hits(self, det_name, det_type='calorimeter', n_copy=1, pid=11):
        try:
            det = getattr(self, det_name)
        except AttributeError:
            return None

        is_primary = det.PTID.content == 0
        pid_correct = det.PID.content == pid
        is_good = is_primary & pid_correct

        if det_type == 'tracking':
            found = []
            fired_did_list = numpy.split(det.DID.content, det.DID.stops)[:-1]
            for copyid in range(n_copy):
                found_in_copy = []
                for i in range(len(fired_did_list)):
                    index_list = det.DID.starts[i] + numpy.nonzero(fired_did_list[i] == copyid)[0]  # 0 means x axis
                    found_in_copy.append(index_list[is_good[index_list]][0] if any(is_good[index_list]) else -1)
                found.append(numpy.array(found_in_copy))
            return found
        elif det_type == 'standard' or det_type == 'calorimeter':
            found = []
            primary_list = numpy.split(is_primary, det.PTID.stops)[:-1]
            for i in range(len(primary_list)):
                index_list = det.PTID.starts[i] + numpy.nonzero(primary_list[i])[0]  # 0 means x axis
                found.append(index_list[is_good[index_list]][0] if any(is_good[index_list]) else -1)
            return numpy.array(found)

        return None
