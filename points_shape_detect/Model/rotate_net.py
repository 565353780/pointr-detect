#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn

from udf_generate.Method.udfs import getPointUDF

from points_shape_detect.Method.trans import transPointArray


class RotateNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.trans_dim = 32

        self.euler_angle_decoder = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3, 1), nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(3, 3, 1))
        return

    @torch.no_grad()
    def generateUDFWithGT(self, data):
        origin_point_array = data['predictions']['origin_point_array']
        origin_query_point_array = data['predictions'][
            'origin_query_point_array']
        gt_euler_angle_inv = data['inputs']['euler_angle_inv']

        device = origin_query_point_array.device

        origin_udf_list = []
        origin_query_udf_list = []
        rotate_back_udf_list = []
        rotate_back_query_udf_list = []

        translate = torch.tensor([0.0, 0.0, 0.0],
                                 dtype=torch.float32).to(device)
        scale = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).to(device)
        for i in range(origin_query_point_array.shape[0]):
            origin_points = origin_point_array[i]
            origin_query_points = origin_query_point_array[i]
            euler_angle = gt_euler_angle_inv[i]

            rotate_back_points = transPointArray(origin_points,
                                                 translate,
                                                 euler_angle,
                                                 scale,
                                                 center=translate)
            rotate_back_query_points = transPointArray(origin_query_points,
                                                       translate,
                                                       euler_angle,
                                                       scale,
                                                       center=translate)
            origin_udf = getPointUDF(origin_points)
            origin_query_udf = getPointUDF(origin_query_points)
            rotate_back_udf = getPointUDF(rotate_back_points)
            rotate_back_query_udf = getPointUDF(rotate_back_query_points)

            origin_udf_list.append(origin_udf.unsqueeze(0))
            origin_query_udf_list.append(origin_query_udf.unsqueeze(0))
            rotate_back_udf_list.append(rotate_back_udf.unsqueeze(0))
            rotate_back_query_udf_list.append(
                rotate_back_query_udf.unsqueeze(0))

        origin_udf = torch.cat(origin_udf_list).detach()
        origin_query_udf = torch.cat(origin_query_udf_list).detach()
        rotate_back_udf = torch.cat(rotate_back_udf_list).detact()
        rotate_back_query_udf = torch.cat(rotate_back_query_udf_list).detach()

        data['predictions']['origin_udf'] = origin_udf
        data['predictions']['origin_query_udf'] = origin_query_udf
        data['predictions']['rotate_back_udf'] = origin_udf
        data['predictions']['rotate_back_query_udf'] = origin_query_udf
        return data

    @torch.no_grad()
    def generateUDF(self, data):
        if self.training:
            return generateUDFWithGT(data)

        return data

    def forward(self, data):
        if self.training:
            data = self.generateUDF(data)

        return data
