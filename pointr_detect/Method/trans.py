#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def normalizePointArray(point_array):
    centroid = np.mean(point_array, axis=0)
    point_array = point_array - centroid
    m = np.max(np.sqrt(np.sum(point_array**2, axis=1)))
    point_array = point_array / m
    return point_array


def moveToOrigin(point_array):
    min_xyz = np.array([np.min(point_array[:, i]) for i in range(3)])
    max_xyz = np.array([np.max(point_array[:, i]) for i in range(3)])

    mean_xyz = (min_xyz + max_xyz) / 2.0

    return point_array - mean_xyz


def moveToMeanPoint(point_array):
    mean_xyz = np.array([np.mean(point_array[:, i]) for i in range(3)])
    return point_array - mean_xyz
