#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn

from points_shape_detect.Method.trans import getInverseTrans, transPointArray
from points_shape_detect.Method.weight import setWeight

from points_shape_detect.Model.bbox_net import BBoxNet
from points_shape_detect.Model.rotate_net import RotateNet


class PointsShapeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.bbox_net = BBoxNet()
        self.rotate_net = RotateNet()
        return

    def addWeight(self, data):
        if not self.training:
            return data

        setWeight(data, 'loss_origin_euler_angle_inv', 1)
        setWeight(data, 'loss_origin_query_euler_angle_inv', 1)
        setWeight(data, 'loss_partial_complete_euler_angle_inv_diff', 1)

        setWeight(data, 'loss_decode_origin_udf', 1000)
        setWeight(data, 'loss_decode_origin_query_udf', 1000)
        return data

    def forward(self, data):
        data = self.bbox_net(data)
        data = self.rotate_net(data)
        return data
