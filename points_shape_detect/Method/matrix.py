#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import quaternion


def make_M_from_tqs(t: list, q: list, s: list, center=None) -> np.ndarray:
    if not isinstance(q, np.quaternion):
        q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    C = np.eye(4)
    if center is not None:
        C[0:3, 3] = center

    M = T.dot(R).dot(S).dot(C)
    return M


def getXRotateMatrix(rotate_angle):
    rotate_rad = rotate_angle * np.pi / 180.0

    x_rotate_matrix = np.array([[1.0, 0.0, 0.0],
                                [0.0,
                                 np.cos(rotate_rad), -np.sin(rotate_rad)],
                                [0.0,
                                 np.sin(rotate_rad),
                                 np.cos(rotate_rad)]])
    return x_rotate_matrix


def getYRotateMatrix(rotate_angle):
    rotate_rad = rotate_angle * np.pi / 180.0

    y_rotate_matrix = np.array([[np.cos(rotate_rad), 0.0,
                                 np.sin(rotate_rad)], [0.0, 1.0, 0.0],
                                [-np.sin(rotate_rad), 0.0,
                                 np.cos(rotate_rad)]])
    return y_rotate_matrix


def getZRotateMatrix(rotate_angle):
    rotate_rad = rotate_angle * np.pi / 180.0

    z_rotate_matrix = np.array([[np.cos(rotate_rad), -np.sin(rotate_rad), 0.0],
                                [np.sin(rotate_rad),
                                 np.cos(rotate_rad), 0.0], [0.0, 0.0, 1.0]])
    return z_rotate_matrix


def getRotateMatrix(xyz_rotate_angle, is_inverse=False):
    x_rotate_matrix = getXRotateMatrix(xyz_rotate_angle[0])
    y_rotate_matrix = getYRotateMatrix(xyz_rotate_angle[1])
    z_rotate_matrix = getZRotateMatrix(xyz_rotate_angle[2])

    if is_inverse:
        return z_rotate_matrix @ y_rotate_matrix @ x_rotate_matrix
    return x_rotate_matrix @ y_rotate_matrix @ z_rotate_matrix
