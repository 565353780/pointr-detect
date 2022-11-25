#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn

from points_shape_detect.Model.encode.points_encoder import PointsEncoder
from points_shape_detect.Model.bbox.bbox_net import BBoxNet
from points_shape_detect.Model.complete.shape_complete_net import ShapeCompleteNet


class PointsShapeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.points_encoder = PointsEncoder()
        self.bbox_net = BBoxNet()
        self.shape_complete_net = ShapeCompleteNet()
        return

    def forward(self, data):
        data = self.points_encoder(data)
        data = self.bbox_net(data)
        data = self.shape_complete_net(data)
        return data
