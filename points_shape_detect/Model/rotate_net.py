#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn

from udf_generate.Method.udfs import getPointUDF

from points_shape_detect.Model.resnet_encoder import ResNetEncoder
from points_shape_detect.Model.resnet_decoder import ResNetDecoder

from points_shape_detect.Method.trans import transPointArray


class RotateNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature_dim = 1024

        self.shape_encoder = ResNetEncoder(self.feature_dim)

        self.euler_angle_encoder = nn.Sequential(
            nn.Conv1d(self.feature_dim, 3, 1),
            nn.LeakyReLU(negative_slope=0.2), nn.Conv1d(3, 3, 1))

        self.shape_decoder = ResNetDecoder(self.shape_encoder.feats)

        self.l1_loss = nn.SmoothL1Loss()
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

            rotate_back_points = transPointArray(origin_points, translate,
                                                 euler_angle, scale, True,
                                                 translate)
            rotate_back_query_points = transPointArray(origin_query_points,
                                                       translate, euler_angle,
                                                       scale, True, translate)
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
        rotate_back_udf = torch.cat(rotate_back_udf_list).detach()
        rotate_back_query_udf = torch.cat(rotate_back_query_udf_list).detach()

        data['predictions']['origin_udf'] = origin_udf
        data['predictions']['origin_query_udf'] = origin_query_udf
        data['predictions']['rotate_back_udf'] = rotate_back_udf
        data['predictions']['rotate_back_query_udf'] = rotate_back_query_udf
        return data

    @torch.no_grad()
    def generateUDF(self, data):
        if self.training:
            return self.generateUDFWithGT(data)

        origin_query_point_array = data['predictions'][
            'origin_query_point_array']

        device = origin_query_point_array.device

        origin_query_udf_list = []

        for i in range(origin_query_point_array.shape[0]):
            origin_query_udf = getPointUDF(origin_query_points)

            origin_query_udf_list.append(origin_query_udf.unsqueeze(0))

        origin_query_udf = torch.cat(origin_query_udf_list).detach()

        data['predictions']['origin_query_udf'] = origin_query_udf
        return data

    def encodeQueryUDFWithGT(self, data):
        # Bx32x32
        origin_udf = data['predictions']['origin_udf']
        # Bx32x32
        origin_query_udf = data['predictions']['origin_query_udf']

        origin_shape_code = self.shape_encoder(origin_udf.unsqueeze(0))
        origin_query_shape_code = self.shape_encoder(
            origin_query_udf.unsqueeze(0))

        data['predictions']['origin_shape_code'] = origin_shape_code
        data['predictions'][
            'origin_query_shape_code'] = origin_query_shape_code
        return data

    def encodeQueryUDF(self, data):
        if self.training:
            return self.encodeQueryUDFWithGT(data)

        # Bx32x32
        origin_query_udf = data['predictions']['origin_query_udf']

        origin_query_shape_code = self.shape_encoder(origin_query_udf)

        data['predictions'][
            'origin_query_shape_code'] = origin_query_shape_code
        return data

    def encodeRotateWithGT(self, data):
        # Bxself.feature_dim
        origin_shape_code = data['predictions']['origin_shape_code']
        # Bxself.feature_dim
        origin_query_shape_code = data['predictions'][
            'origin_query_shape_code']

        B, C = data['predictions']['origin_query_shape_code'].shape

        origin_euler_angle_inv = self.euler_angle_encoder(
            origin_shape_code.unsqueeze(-1)).reshape(B, -1)
        origin_query_euler_angle_inv = self.euler_angle_encoder(
            origin_query_shape_code.unsqueeze(-1)).reshape(B, -1)

        data['predictions']['origin_euler_angle_inv'] = origin_euler_angle_inv
        data['predictions'][
            'origin_query_euler_angle_inv'] = origin_query_euler_angle_inv

        if self.training:
            data = self.lossRotate(data)
        return data

    def encodeRotate(self, data):
        if self.training:
            return self.encodeRotateWithGT(data)

        # Bxself.feature_dim
        origin_query_shape_code = data['predictions'][
            'origin_query_shape_code']

        B, C = data['predictions']['origin_query_shape_code'].shape

        origin_query_euler_angle_inv = self.euler_angle_encoder(
            origin_query_shape_code.unsqueeze(-1)).reshape(B, -1)

        data['predictions'][
            'origin_query_euler_angle_inv'] = origin_query_euler_angle_inv
        return data

    def lossRotate(self, data):
        origin_euler_angle_inv = data['predictions']['origin_euler_angle_inv']
        origin_query_euler_angle_inv = data['predictions'][
            'origin_query_euler_angle_inv']
        gt_euler_angle_inv = data['inputs']['euler_angle_inv']

        loss_origin_euler_angle_inv = self.l1_loss(origin_euler_angle_inv,
                                                   gt_euler_angle_inv)
        loss_origin_query_euler_angle_inv = self.l1_loss(
            origin_query_euler_angle_inv, gt_euler_angle_inv)
        loss_partial_complete_euler_angle_inv_diff = self.l1_loss(
            origin_euler_angle_inv, origin_query_euler_angle_inv)

        data['losses'][
            'loss_origin_euler_angle_inv'] = loss_origin_euler_angle_inv
        data['losses'][
            'loss_origin_query_euler_angle_inv'] = loss_origin_query_euler_angle_inv
        data['losses'][
            'loss_partial_complete_euler_angle_inv_diff'] = loss_partial_complete_euler_angle_inv_diff
        return data

    def decodeShapeCodeWithGT(self, data):
        origin_shape_code = data['predictions']['origin_shape_code']
        origin_query_shape_code = data['predictions'][
            'origin_query_shape_code']

        decode_origin_udf = self.shape_decoder(origin_shape_code)
        decode_origin_query_udf = self.shape_decoder(origin_query_shape_code)

        data['predictions']['decode_origin_udf'] = decode_origin_udf
        data['predictions'][
            'decode_origin_query_udf'] = decode_origin_query_udf

        self.lossUDF(data)
        return data

    def decodeShapeCode(self, data):
        if self.training:
            return self.decodeShapeCodeWithGT(data)

        origin_query_shape_code = data['predictions'][
            'origin_query_shape_code']

        decode_origin_query_udf = self.shape_decoder(origin_query_shape_code)

        data['predictions'][
            'decode_origin_query_udf'] = decode_origin_query_udf
        return data

    def lossUDF(self, data):
        rotate_back_udf = data['predictions']['rotate_back_udf']
        rotate_back_query_udf = data['predictions']['rotate_back_query_udf']
        decode_origin_udf = data['predictions']['decode_origin_udf']
        decode_origin_query_udf = data['predictions'][
            'decode_origin_query_udf']

        loss_decode_origin_udf = self.l1_loss(decode_origin_udf,
                                              rotate_back_udf)
        loss_decode_origin_query_udf = self.l1_loss(decode_origin_query_udf,
                                                    rotate_back_query_udf)

        data['losses']['loss_decode_origin_udf'] = loss_decode_origin_udf
        data['losses'][
            'loss_decode_origin_query_udf'] = loss_decode_origin_query_udf
        return data

    def forward(self, data):
        data = self.generateUDF(data)

        data = self.encodeQueryUDF(data)

        data = self.encodeRotate(data)

        data = self.decodeShapeCode(data)
        return data
