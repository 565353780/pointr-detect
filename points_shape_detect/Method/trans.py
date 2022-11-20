#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from points_shape_detect.Method.matrix import make_M_from_tqs, decompose_mat4


def normalizePointArray(point_array):
    #  centroid = np.mean(point_array, axis=0)
    min_point = np.min(point_array, axis=0)
    max_point = np.max(point_array, axis=0)
    centroid = np.mean([min_point, max_point], axis=0)

    point_array = point_array - centroid

    #  m = np.max(np.sqrt(np.sum(point_array**2, axis=1)))
    #  point_array = point_array / m
    max_bbox_length = np.max(max_point - min_point)
    point_array = point_array / max_bbox_length
    return point_array


def getQuatFromEulerAngle(euler_angle):
    r = R.from_euler('zxy', euler_angle, degrees=True)
    return r.as_quat()


def getRotateMatrixFromEulerAngle(euler_angle):
    # yaw, roll, pitch
    r = R.from_euler('zxy', euler_angle, degrees=True)
    return r.as_matrix()


def getInverseTrans(translate, quat, scale):
    trans_matrix = make_M_from_tqs(translate, quat, scale)
    trans_matrix_inv = np.linalg.inv(trans_matrix)
    translate_inv, quat_inv, scale_inv = decompose_mat4(trans_matrix_inv)
    return translate_inv, quat_inv, scale_inv


def transPointArray(point_array, translate, quat, scale):
    trans_matrix = make_M_from_tqs(translate, quat, scale)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_array)
    pcd.transform(trans_matrix)

    trans_point_array = np.array(pcd.points)
    return trans_point_array


def randomTransPointArray(point_array, need_trans=False):
    translate = np.random.rand(3) - 0.5
    euler_angle = (np.random.rand(3) - 0.5) * 360.0
    scale = np.random.rand(3) + 0.5

    quat = getQuatFromEulerAngle(euler_angle)
    trans_point_array = transPointArray(point_array, translate, quat, scale)
    if need_trans:
        return trans_point_array, translate, quat, scale
    return trans_point_array


def moveToOrigin(point_array):
    min_xyz = np.array([np.min(point_array[:, i]) for i in range(3)])
    max_xyz = np.array([np.max(point_array[:, i]) for i in range(3)])

    mean_xyz = (min_xyz + max_xyz) / 2.0

    return point_array - mean_xyz


def moveToMeanPoint(point_array):
    mean_xyz = np.array([np.mean(point_array[:, i]) for i in range(3)])
    return point_array - mean_xyz
