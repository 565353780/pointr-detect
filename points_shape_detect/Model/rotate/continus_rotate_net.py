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
        super(Model, self).__init__()

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

        rotate_back_points_list = []
        rotate_back_query_points_list = []

        for i in range(origin_query_point_array.shape[0]):
            origin_points = origin_point_array[i]
            origin_query_points = origin_query_point_array[i]
            rotate_matrix_inv = rotate_matrix[i].transpose()

            rotate_back_points = torch.matmul(origin_points, rotate_matrix_inv)
            rotate_back_query_points = torch.matmul(origin_query_points,
                                                    rotate_matrix_inv)

            rotate_back_points_list.append(rotate_back_points.unsqueeze(0))
            rotate_back_query_points_list.append(
                rotate_back_query_points.unsqueeze(0))

        rotate_back_point_array = torch.cat(rotate_back_points_list).detach()
        rotate_back_query_point_array = torch.cat(
            rotate_back_query_points_list).detach()

        data['inputs']['rotate_back_point_array'] = rotate_back_point_array
        data['inputs'][
            'rotate_back_query_point_array'] = rotate_back_query_point_array
        return data

    def encodeRotationMatrix(self, data):
        # b*N*3
        pt1 = data['inputs']['rotate_back_origin_query_point_array']
        # b*N*3
        pt2 = data['predictions']['origin_query_point_array']

        B, N, _ = pt1.shape

        # bx1024
        feature_pt1 = self.feature_extracter(pt1.transpose(1, 2)).view(B, -1)
        # bx1024
        feature_pt2 = self.feature_extracter(pt2.transpose(1, 2)).view(B, -1)

        # bx2048
        f = torch.cat((feature_pt1, feature_pt2), 1)

        # bx6
        rotation = self.mlp(f)

        # bx3x3
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
        loss_geodesic = compute_geodesic_distance_from_two_matrices(
            gt_rotate_matrix, rotate_matrix).mean()

        data['losses']['loss_rotate_matrix'] = loss_rotate_matrix
        data['losses']['loss_geodesic'] = loss_geodesic
        return data

    def rotateBackByPredict(self, data):
        pt1 = data['predictions']['origin_point_array']
        pt2 = data['predictions']['origin_query_point_array']
        rotate_matrix = data['predictions']['rotate_matrix']

        rotate_matrix_inv = rotate_matrix.transpose(1, 2)

        B, N, _ = pt2.shape

        rotate_back_point_array = torch.bmm(
            rotate_matrix_inv.view(B, 1, 3,
                                   3).expand(B, N, 3,
                                             3).contiguous().view(-1, 3, 3),
            pt1.view(-1, 3, 1)).view(B, N, 3)

        rotate_back_query_point_array = torch.bmm(
            rotate_matrix_inv.view(B, 1, 3,
                                   3).expand(B, N, 3,
                                             3).contiguous().view(-1, 3, 3),
            pt2.view(-1, 3, 1)).view(B, N, 3)

        data['predictions'][
            'rotate_back_point_array'] = rotate_back_point_array
        data['predictions'][
            'rotate_back_query_point_array'] = rotate_back_query_point_array

        #  if self.training:
        data = self.lossPose(data)
        return data

    def lossPose(self, data):
        rotate_back_point_array = data['predictions'][
            'rotate_back_point_array']
        rotate_back_query_point_array = data['predictions'][
            'rotate_back_query_point_array']
        gt_rotate_back_point_array = data['inputs']['rotate_back_point_array']
        gt_rotate_back_query_point_array = data['inputs'][
            'rotate_back_query_point_array']

        loss_complete_pose = self.mse_loss(rotate_back_point_array,
                                           gt_rotate_back_point_array)
        loss_query_pose = self.mse_loss(rotate_back_query_point_array,
                                        gt_rotate_back_query_point_array)

        data['losses']['loss_complete_pose'] = loss_complete_pose
        data['losses']['loss_query_pose'] = loss_query_pose
        return data

    def addWeight(self, data):
        #  if not self.training:
        #  return data

        data = setWeight(data, 'loss_rotate_matrix', 1)
        data = setWeight(data, 'loss_geodesic', 1)

        data = setWeight(data, 'loss_complete_pose', 1)
        data = setWeight(data, 'loss_query_pose', 1)
        return data

    def forward(self, data):
        data = self.rotateBackPoints(data)

        data = self.encodeRotation(data)

        data = self.rotateBackByPredict(data)

        data = self.addWeight(data)
        return data
