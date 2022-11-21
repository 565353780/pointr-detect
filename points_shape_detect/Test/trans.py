#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from tqdm import tqdm

from points_shape_detect.Method.trans import \
    normalizePointArray, randomTransPointArray, \
    getInverseTrans, transPointArray
from points_shape_detect.Method.render import renderPointArrayWithUnitBBox

from points_shape_detect.Method.matrix import getRotateMatrix


def testMatrix(print_progress=False):
    test_num = 10000

    error = 1e-10

    for_data = range(test_num)
    if print_progress:
        for_data = tqdm(for_data)
    for _ in for_data:
        euler_angle = (np.random.rand(3) - 0.5) * 360.0
        euler_angle_inv = -1.0 * euler_angle
        rotate_matrix = getRotateMatrix(euler_angle)
        rotate_matrix_inv = getRotateMatrix(euler_angle_inv, True)
        rotate_matrix_inv2 = np.linalg.inv(rotate_matrix_inv)
        assert np.linalg.norm(rotate_matrix - rotate_matrix_inv2) < error

        npy_file_path = "/home/chli/chLi/PoinTr/ShapeNet55/shapenet_pc/" + \
            "04090263-2eb7e88f5355630962a5697e98a94be.npy"
        point_array = np.load(npy_file_path)

        center = np.mean(point_array, axis=0)

        origin_point_array = point_array - center

        trans_point_array = origin_point_array @ rotate_matrix

        trans_back_point_array = trans_point_array @ rotate_matrix_inv

        move_back_point_array = trans_back_point_array + center

        renderPointArrayWithUnitBBox(
            np.vstack(
                (point_array, move_back_point_array + np.array([0, 0, 1]))))
    return True


def testEuler(print_progress=False):
    test_num = 100

    error = 1e-2

    for_data = range(test_num)
    if print_progress:
        for_data = tqdm(for_data)
    for _ in for_data:

        npy_file_path = "/home/chli/chLi/PoinTr/ShapeNet55/shapenet_pc/" + \
            "04090263-2eb7e88f5355630962a5697e98a94be.npy"
        point_array = np.load(npy_file_path)

        #  point_array = np.random.randn(8192, 3)

        normalize_point_array = normalizePointArray(point_array)

        random_point_array, translate, euler_angle, scale = randomTransPointArray(
            normalize_point_array, True)

        translate_inv, euler_angle_inv, scale_inv = getInverseTrans(
            translate, euler_angle, scale)
        move_back_point_array = transPointArray(random_point_array,
                                                translate_inv, euler_angle_inv,
                                                scale_inv, True)

        translate_inv2, euler_angle_inv2, scale_inv2 = getInverseTrans(
            translate_inv, euler_angle_inv, scale_inv)

        assert np.linalg.norm(translate_inv2 - translate) < error
        assert np.linalg.norm(euler_angle_inv2 - euler_angle) < error
        assert np.linalg.norm(scale_inv2 - scale) < error

        print(np.linalg.norm(move_back_point_array - normalize_point_array),
              error * point_array.shape[0])
        renderPointArrayWithUnitBBox(
            np.vstack((normalize_point_array,
                       move_back_point_array + np.array([0, 0, 1]))))

        assert np.linalg.norm(move_back_point_array - normalize_point_array
                              ) < error * point_array.shape[0]
    return True


def testEulerTensor(print_progress=False):
    test_num = 100

    error = 1e-2

    for_data = range(test_num)
    if print_progress:
        for_data = tqdm(for_data)
    for _ in for_data:
        point_array = np.random.randn(8192, 3)
        point_array_tensor = torch.from_numpy(point_array)
        normalize_point_array_tensor = normalizePointArray(point_array_tensor)

        random_point_array, translate, euler_angle, scale = randomTransPointArray(
            normalize_point_array, True)

        translate_inv, euler_angle_inv, scale_inv = getInverseTrans(
            translate, euler_angle, scale)
        move_back_point_array = transPointArray(random_point_array,
                                                translate_inv, euler_angle_inv,
                                                scale_inv, True)

        translate_inv2, euler_angle_inv2, scale_inv2 = getInverseTrans(
            translate_inv, euler_angle_inv, scale_inv)

        assert np.linalg.norm(translate_inv2 - translate) < error
        assert np.linalg.norm(euler_angle_inv2 - euler_angle) < error
        assert np.linalg.norm(scale_inv2 - scale) < error

        #  renderPointArrayWithUnitBBox(
        #  np.vstack((normalize_point_array,
        #  move_back_point_array + np.array([0, 0, 1]))))

        assert np.linalg.norm(move_back_point_array - normalize_point_array
                              ) < error * point_array.shape[0]
    return True


def test():
    print_progress = False

    print("[INFO][trans::test] start testMatrix...")
    assert testMatrix(print_progress)
    print("\t passed!")

    print("[INFO][trans::test] start testEuler...")
    assert testEuler(print_progress)
    print("\t passed!")

    print("[INFO][trans::test] start testEulerTensor...")
    assert testEulerTensor(print_progress)
    print("\t passed!")
    return True
