import datetime
import math
import os
import pickle
from distutils.util import strtobool

import numpy as np


def timestamp(divider='-', datetime_divider='T'):
    now = datetime.datetime.now()
    return now.strftime(
        '%Y{d}%m{d}%dT%H{d}%M{d}%S'
        ''.format(d=divider, dtd=datetime_divider))


def str2bool(x):
    return bool(strtobool(x))

def angle(p0, p1):
    v0 = np.array(p0) - np.array(p1)

    angle_radians = math.atan2(v0[1], v0[0])
    while angle_radians < 0:
        angle_radians += 2 * math.pi
    while angle_radians > 2 * math.pi:
        angle_radians -= 2 * math.pi
    return angle_radians

def true_angle_diff(theta):
    """theta is before the absolute value is applied"""
    print("abs(theta)", abs(theta))
    print("abs(theta - 2 * np.pi)", abs(theta - 2 * np.pi))
    return min(abs(theta), abs(theta - 2 * np.pi))

def copysign(a, b):
    a = np.array(a).repeat(b.shape[0])
    return np.abs(a) * np.sign(b)


def quat_to_deg(q):
    q = np.array(q)
    q = q[None]

    return quat_to_deg_batch(q)[0]


def quat_to_deg_batch(q):
    qx, qy, qz, qw = [0], [1], [2], [3]
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    # roll = np.arctan2(sinr_cosp, cosr_cosp) % (2 * np.pi)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    # pitch = np.where(np.abs(sinp) >= 1, copysign(
    #     np.pi / 2.0, sinp), np.arcsin(sinp)) % (2 * np.pi)
    pitch = np.where(np.abs(sinp) >= 1, copysign(
        np.pi / 2.0, sinp), np.arcsin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    # yaw = np.arctan2(siny_cosp, cosy_cosp) % (2 * np.pi)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.concatenate([roll, pitch, yaw], axis=-1) * 180 / np.pi


def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

class DemoPool:

    def __init__(self, max_size=1e6):
        self._keys = ('observations', 'actions', 'next_observations', 'rewards', 'terminals')
        self._fields = {}
        self._max_size = int(max_size)
        self._size = 0
        self._pointer = 0

    @property
    def size(self):
        return self._size

    def add_sample(self, *arrays):
        if self._size:
            self._add(arrays)
        else:
            self._init(arrays)

        self._advance()
        # print(self._size, self._pointer)

    def save(self, params, *savepath):
        savepath = os.path.join(*savepath)
        self._prune()
        save_info = [(key, self._fields[key].shape) for key in self._keys]
        print('[ DemoPool ] Saving to: {} | {}'.format(savepath, save_info))
        pickle.dump(self._fields, open(savepath, 'wb+'))

        ## save params
        params_path = savepath.replace('pool', 'params')
        pickle.dump(params, open(params_path, 'wb+'))

    def _add(self, arrays):
        for key, array in zip(self._keys, arrays):
            self._fields[key][self._pointer] = array

    def _init(self, arrays):
        for key, array in zip(self._keys, arrays):
            shape = array.shape if type(array) == np.ndarray else (1,)
            dtype = array.dtype if type(array) == np.ndarray else type(array)
            self._fields[key] = np.zeros((self._max_size, *shape), dtype=dtype)
            self._fields[key][self._pointer] = array
            # print(key, self._fields[key].shape, self._fields[key].dtype)

    def _advance(self):
        self._size = min(self._size + 1, self._max_size)
        self._pointer = (self._pointer + 1) % self._max_size

    def _prune(self):
        for key in self._keys:
            self._fields[key] = self._fields[key][:self._size]

    def get_samples(self):
        self._prune()
        return self._fields


class Meta:

    def __init__(self, fn, *args, **kwargs):
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def __call__(self, *args, **kwargs):
        self._kwargs.update(**kwargs)
        return self._fn(*args, *self._args, **self._kwargs)

