#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn

from points_shape_detect.Method.trans import getInverseTrans, transPointArray

from points_shape_detect.Model.encode.points_encoder import PointsEncoder
from points_shape_detect.Model.bbox.bbox_net import BBoxNet
from points_shape_detect.Model.rotate.continus_rotate_net import ContinusRotateNet
from points_shape_detect.Model.complete.shape_complete_net import ShapeCompleteNet


class PointsShapeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.points_encoder = PointsEncoder()
        self.bbox_net = BBoxNet()
        self.continus_rotate_net = ContinusRotateNet()
        self.shape_complete_net = ShapeCompleteNet()
        return

    def forward(self, data):
        data = self.points_encoder(data)
        data = self.bbox_net(data)
        data = self.continus_rotate_net(data)
        data = self.shape_complete_net(data)
        return data
