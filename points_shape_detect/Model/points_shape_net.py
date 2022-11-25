#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn

from points_shape_detect.Method.trans import getInverseTrans, transPointArray

from points_shape_detect.Model.encode.points_encoder import PointsEncoder
from points_shape_detect.Model.bbox.bbox_net import BBoxNet
from points_shape_detect.Model.rotate.coarse_rotate_net import CoarseRotateNet
from points_shape_detect.Model.complete.shape_complete_net import ShapeCompleteNet


class PointsShapeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.points_encoder = PointsEncoder()
        self.bbox_net = BBoxNet()
        self.coarse_rotate_net = CoarseRotateNet()
        self.shape_complete_net = ShapeCompleteNet()
        return

    @torch.no_grad()
    def rotateBackPoints(self, data):
        origin_point_array = data['predictions']['origin_point_array']
        # Bx#pointx3
        origin_query_point_array = data['predictions'][
            'origin_query_point_array']
        euler_angle = data['predictions']['origin_query_euler_angle']

        device = origin_query_point_array.device

        rotate_back_points_list = []
        rotate_back_query_points_list = []

        translate = torch.tensor([0.0, 0.0, 0.0],
                                 dtype=torch.float32).to(device)
        scale = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).to(device)
        for i in range(origin_query_point_array.shape[0]):
            origin_points = origin_point_array[i]
            origin_query_points = origin_query_point_array[i]
            current_euler_angle = euler_angle[i]
            _, current_euler_angle_inv, _ = getInverseTrans(
                translate, current_euler_angle, scale)

            rotate_back_points = transPointArray(origin_points, translate,
                                                 current_euler_angle_inv,
                                                 scale, True, translate)
            rotate_back_query_points = transPointArray(
                origin_query_points, translate, current_euler_angle_inv, scale,
                True, translate)

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
        data = self.points_encoder(data)
        data = self.bbox_net(data)
        data = self.coarse_rotate_net(data)
        data = self.shape_complete_net(data)
        return data
