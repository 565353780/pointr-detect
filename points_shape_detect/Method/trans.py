#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.transform import Rotation as R


def normalizePointArray(point_array):
    #  centroid = np.mean(point_array, axis=0)
    min_point = np.min(point_array, axis=0)
    max_point = np.max(point_array, axis=0)
    centroid = np.mean([min_point, max_point], axis=0)

    point_array = point_array - centroid
    m = np.max(np.sqrt(np.sum(point_array**2, axis=1)))
    point_array = point_array / m
    return point_array


def getRotateMatrixFromEulerAngle(euler_angle):
    # yaw, roll, pitch
    r = R.from_euler('zxy', euler_angle, degrees=True)
    return r.as_matrix()


def randomTransPointArray(point_array):
    point_array = normalizePointArray(point_array)

    random_euler_angle = (np.random.rand(3) - 0.5) * 360.0
    rotate_matrix = getRotateMatrixFromEulerAngle(random_euler_angle)
    point_array = point_array @ rotate_matrix

    random_scale = np.random.rand() + 0.5
    point_array = point_array * random_scale

    random_translate = np.random.rand(3) - 0.5
    point_array = point_array + random_translate
    return point_array


def moveToOrigin(point_array):
    min_xyz = np.array([np.min(point_array[:, i]) for i in range(3)])
    max_xyz = np.array([np.max(point_array[:, i]) for i in range(3)])

    mean_xyz = (min_xyz + max_xyz) / 2.0

    return point_array - mean_xyz


def moveToMeanPoint(point_array):
    mean_xyz = np.array([np.mean(point_array[:, i]) for i in range(3)])
    return point_array - mean_xyz
