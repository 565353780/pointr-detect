#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn

from points_shape_detect.Method.rotate import (
    compute_geodesic_distance_from_two_matrices,
    compute_rotation_matrix_from_ortho6d)
from points_shape_detect.Method.weight import setWeight


class ContinusRotateNet(nn.Module):

    def __init__(self):
        super().__init__()

        # bx#pointx3 -> bx1x1024
        self.feature_extracter = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1), nn.LeakyReLU(),
            nn.Conv1d(64, 128, kernel_size=1), nn.LeakyReLU(),
            nn.Conv1d(128, 1024, kernel_size=1),
            nn.AdaptiveMaxPool1d(output_size=1))

        # bx2048 -> bx6
        self.mlp = nn.Sequential(nn.Linear(2048, 512), nn.LeakyReLU(),
                                 nn.Linear(512, 6))

        self.mse_loss = nn.MSELoss()
        return

    @torch.no_grad()
    def rotateBackPoints(self, data):
        origin_point_array = data['inputs']['origin_point_array']
        # Bx#pointx3
        origin_query_point_array = data['inputs']['origin_query_point_array']
        rotate_matrix = data['inputs']['rotate_matrix']

        rotate_matrix_inv = rotate_matrix.transpose(1, 2)

        rotate_back_point_array = torch.matmul(origin_point_array,
                                               rotate_matrix_inv).detach()
        rotate_back_query_point_array = torch.matmul(
            origin_query_point_array, rotate_matrix_inv).detach()

        data['inputs']['rotate_back_point_array'] = rotate_back_point_array
        data['inputs'][
            'rotate_back_query_point_array'] = rotate_back_query_point_array
        return data

    def encodeRotateMatrix(self, data):
        # B*N*3
        pt1 = data['inputs']['rotate_back_point_array']
        # B*N*3
        pt2 = data['inputs']['origin_point_array']

        B, N, _ = pt1.shape

        # Bx1024
        feature_pt1 = self.feature_extracter(pt1.transpose(1, 2)).view(B, -1)
        # Bx1024
        feature_pt2 = self.feature_extracter(pt2.transpose(1, 2)).view(B, -1)

        # Bx2048
        f = torch.cat((feature_pt1, feature_pt2), 1)

        # Bx6
        rotation = self.mlp(f)

        # Bx3x3
        rotate_matrix = compute_rotation_matrix_from_ortho6d(rotation)

        data['predictions']['rotation'] = rotation
        data['predictions']['rotate_matrix'] = rotate_matrix

        #  if self.training:
        data = self.lossRotate(data)
        return data

    def lossRotate(self, data):
        rotate_matrix = data['predictions']['rotate_matrix']
        gt_rotate_matrix = data['inputs']['rotate_matrix']

        loss_rotate_matrix = torch.pow(gt_rotate_matrix - rotate_matrix,
                                       2).mean()
        #  loss_geodesic = compute_geodesic_distance_from_two_matrices(
        #  gt_rotate_matrix, rotate_matrix).mean()

        data['losses']['loss_rotate_matrix'] = loss_rotate_matrix
        #  data['losses']['loss_geodesic'] = loss_geodesic
        return data

    @torch.no_grad()
    def rotateBackByPredict(self, data):
        pt1 = data['inputs']['origin_point_array']
        pt2 = data['inputs']['origin_query_point_array']
        rotate_matrix = data['predictions']['rotate_matrix']

        rotate_matrix_inv = rotate_matrix.transpose(1, 2)

        B, N, _ = pt2.shape

        rotate_back_point_array = torch.bmm(pt1, rotate_matrix_inv).detach()

        rotate_back_query_point_array = torch.bmm(pt2,
                                                  rotate_matrix_inv).detach()

        data['predictions'][
            'rotate_back_point_array'] = rotate_back_point_array
        data['predictions'][
            'rotate_back_query_point_array'] = rotate_back_query_point_array
        return data

    def addWeight(self, data):
        #  if not self.training:
        #  return data

        data = setWeight(data, 'loss_rotate_matrix', 1)
        #  data = setWeight(data, 'loss_geodesic', 1)
        return data

    def forward(self, data):
        data = self.rotateBackPoints(data)

        data = self.encodeRotateMatrix(data)

        #  data = self.rotateBackByPredict(data)

        data = self.addWeight(data)
        return data
