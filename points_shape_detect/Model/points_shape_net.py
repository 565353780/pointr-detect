#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn

from points_shape_detect.Method.trans import getInverseTrans, transPointArray

from points_shape_detect.Model.bbox_net import BBoxNet
from points_shape_detect.Model.rotate_net import RotateNet


class PointsShapeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.bbox_net = BBoxNet()
        self.rotate_net = RotateNet()
        return

    @torch.no_grad()
    def rotateBackPoints(self, data):
        origin_point_array = data['predictions']['origin_point_array']
        # Bx#pointx3
        origin_query_point_array = data['predictions'][
            'origin_query_point_array']
        euler_angle_inv = data['predictions']['origin_query_euler_angle_inv']

        device = origin_query_point_array.device

        rotate_back_points_list = []
        rotate_back_query_points_list = []

        translate = torch.tensor([0.0, 0.0, 0.0],
                                 dtype=torch.float32).to(device)
        scale = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).to(device)
        for i in range(origin_query_point_array.shape[0]):
            origin_points = origin_point_array[i]
            origin_query_points = origin_query_point_array[i]
            euler_angle = euler_angle_inv[i]

            rotate_back_points = transPointArray(origin_points, translate,
                                                 euler_angle, scale, True,
                                                 translate)
            rotate_back_query_points = transPointArray(origin_query_points,
                                                       translate, euler_angle,
                                                       scale, True, translate)

            rotate_back_points_list.append(rotate_back_points.unsqueeze(0))
            rotate_back_query_points_list.append(
                rotate_back_query_points.unsqueeze(0))

        rotate_back_point_array = torch.cat(rotate_back_points_list).detach()
        rotate_back_query_point_array = torch.cat(
            rotate_back_query_points_list).detach()

        data['predictions'][
            'rotate_back_point_array'] = rotate_back_point_array
        data['predictions'][
            'rotate_back_query_point_array'] = rotate_back_query_point_array
        return data

    def forward(self, data):
        data = self.bbox_net(data)
        data = self.rotate_net(data)
        return data
