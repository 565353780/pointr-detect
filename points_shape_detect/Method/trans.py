#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np

from points_shape_detect.Method.matrix import getRotateMatrix


def getBatchResult(func, batch_data):
    batch_result = torch.cat(
        [func(batch_data[i]).unsqueeze(0) for i in range(batch_data.shape[0])])
    return batch_result


def normalizePointArrayTensor(point_array_tensor):
    min_point_tensor = torch.min(point_array_tensor, 0)[0]
    max_point_tensor = torch.max(point_array_tensor, 0)[0]
    min_max_point_tensor = torch.cat([min_point_tensor,
                                      max_point_tensor]).reshape(2, 3)
    center = torch.mean(min_max_point_tensor, axis=0)

    origin_point_array_tensor = point_array_tensor - center

    max_bbox_length = torch.max(max_point_tensor - min_point_tensor)
    normalize_point_array_tensor = origin_point_array_tensor / max_bbox_length
    return normalize_point_array_tensor


def normalizePointArray(point_array):
    if isinstance(point_array, torch.Tensor):
        assert 2 <= len(point_array.shape) <= 3
        if len(point_array.shape) == 2:
            return normalizePointArrayTensor(point_array)
        else:
            return getBatchResult(normalizePointArrayTensor, point_array)

    min_point = np.min(point_array, axis=0)
    max_point = np.max(point_array, axis=0)
    center = np.mean([min_point, max_point], axis=0)

    origin_point_array = point_array - center

    max_bbox_length = np.max(max_point - min_point)
    normalize_point_array = origin_point_array / max_bbox_length
    return normalize_point_array


def getInverseTrans(translate, euler_angle, scale):
    translate_inv = -1.0 * translate
    euler_angle_inv = -1.0 * euler_angle
    scale_inv = 1.0 / scale
    return translate_inv, euler_angle_inv, scale_inv


def transPointArrayTensor(point_array_tensor,
                          translate,
                          euler_angle,
                          scale,
                          is_inverse=False):
    min_point_tensor = torch.min(point_array_tensor, 0)[0]
    max_point_tensor = torch.max(point_array_tensor, 0)[0]
    min_max_point_tensor = torch.cat([min_point_tensor,
                                      max_point_tensor]).reshape(2, 3)
    center = torch.mean(min_max_point_tensor, axis=0)

    origin_point_array_tensor = point_array_tensor - center

    rotate_matrix = getRotateMatrix(euler_angle, is_inverse)

    origin_trans_point_array_tensor = torch.matmul(origin_point_array_tensor,
                                                   rotate_matrix)

    trans_point_array = origin_trans_point_array + center + translate
    return trans_point_array


def transPointArray(point_array,
                    translate,
                    euler_angle,
                    scale,
                    is_inverse=False,
                    rotate_center=None):
    if rotate_center is None:
        center = np.mean(point_array, axis=0)
    else:
        center = np.array(rotate_center)

    origin_point_array = point_array - center

    rotate_matrix = getRotateMatrix(euler_angle, is_inverse)

    origin_trans_point_array = origin_point_array @ rotate_matrix

    scale_origin_trans_point_array = origin_trans_point_array * scale

    trans_point_array = scale_origin_trans_point_array + center + translate
    return trans_point_array


def randomTransPointArray(point_array, need_trans=False):
    translate = np.random.rand(3) - 0.5
    euler_angle = (np.random.rand(3) - 0.5) * 360.0
    scale = np.random.rand(3) + 0.5

    trans_point_array = transPointArray(point_array, translate, euler_angle,
                                        scale)
    if need_trans:
        return trans_point_array, translate, euler_angle, scale
    return trans_point_array


def moveToOrigin(point_array):
    min_point = np.min(point_array, axis=0)
    max_point = np.max(point_array, axis=0)
    center = np.mean([min_point, max_point], axis=0)
    return point_array - center


def moveToMeanPoint(point_array):
    mean_xyz = np.array([np.mean(point_array[:, i]) for i in range(3)])
    return point_array - mean_xyz
