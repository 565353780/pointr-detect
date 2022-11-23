#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn

from udf_generate.Method.udfs import getPointUDF

from points_shape_detect.Method.trans import transPointArray


class RotateNet(nn.Module):

    def __init__(self):
        return

    @torch.no_grad()
    def rotateBack(self, data):
        gt_euler_angle_inv = data['inputs']['euler_angle_inv']
        gt_scale_inv = data['inputs']['scale_inv']

        device = origin_trans_query_point_array.device

        return data

    def forward(self, data):
        data = self.rotateBack(data)

        return data
