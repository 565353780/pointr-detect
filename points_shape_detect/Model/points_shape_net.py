#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from pointnet2_ops import pointnet2_utils
from torch import nn

from points_shape_detect.Lib.chamfer_dist import ChamferDistanceL1
from points_shape_detect.Loss.ious import IoULoss
from points_shape_detect.Method.sample import fps
from points_shape_detect.Method.trans import getInverseTrans, transPointArray
from points_shape_detect.Model.fold import Fold
from points_shape_detect.Model.pc_transformer import PCTransformer
from points_shape_detect.Model.rotate_net import RotateNet


class PointsShapeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.trans_dim = 384
        self.knn_layer = 1
        self.num_pred = 6144
        self.num_query = 96

        self.fold_step = int(pow(self.num_pred // self.num_query, 0.5) + 0.5)

        self.rotate_net = RotateNet()

        self.feature_encoder = PCTransformer(in_chans=3,
                                             embed_dim=self.trans_dim,
                                             depth=[6, 8],
                                             drop_rate=0.,
                                             num_query=self.num_query,
                                             knn_layer=self.knn_layer)

        self.foldingnet = Fold(self.trans_dim,
                               step=self.fold_step,
                               hidden_dim=256)  # rebuild a cluster point

        self.increase_dim = nn.Sequential(nn.Conv1d(self.trans_dim, 1024, 1),
                                          nn.BatchNorm1d(1024),
                                          nn.LeakyReLU(negative_slope=0.2),
                                          nn.Conv1d(1024, 1024, 1))

        self.reduce_map = nn.Linear(self.trans_dim + 1027, self.trans_dim)

        self.bbox_feature_encoder = nn.Sequential(
            nn.Conv1d(self.num_query * self.trans_dim, self.trans_dim, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, self.trans_dim, 1))

        self.bbox_decoder = nn.Sequential(nn.Conv1d(self.trans_dim, 6, 1),
                                          nn.LeakyReLU(negative_slope=0.2),
                                          nn.Conv1d(6, 6, 1))

        self.center_decoder = nn.Sequential(nn.Conv1d(self.trans_dim, 3, 1),
                                            nn.LeakyReLU(negative_slope=0.2),
                                            nn.Conv1d(3, 3, 1))

        self.scale_decoder = nn.Sequential(nn.Conv1d(self.trans_dim, 3, 1),
                                           nn.LeakyReLU(negative_slope=0.2),
                                           nn.Conv1d(3, 3, 1))

        self.loss_func = ChamferDistanceL1()
        self.l1_loss = nn.SmoothL1Loss()
        return

    @torch.no_grad()
    def moveToOrigin(self, data):
        trans_point_array = data['inputs']['trans_point_array']
        # Bx#pointx3
        trans_query_point_array = data['inputs']['trans_query_point_array']

        origin_points_list = []
        origin_query_points_list = []

        for i in range(trans_query_point_array.shape[0]):
            trans_points = trans_point_array[i]
            trans_query_points = trans_query_point_array[i]
            trans_query_points_center = torch.mean(trans_query_points, 0)

            origin_points = trans_points - trans_query_points_center
            origin_query_points = trans_query_points - trans_query_points_center

            origin_points_list.append(origin_points.unsqueeze(0))
            origin_query_points_list.append(origin_query_points.unsqueeze(0))

        origin_point_array = torch.cat(origin_points_list).detach()
        origin_query_point_array = torch.cat(origin_query_points_list).detach()

        data['predictions']['origin_point_array'] = origin_point_array
        data['predictions'][
            'origin_query_point_array'] = origin_query_point_array
        return data

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

    def encodeRotateBackPoints(self, data):
        # Bx#pointx3
        rotate_back_query_point_array = data['predictions'][
            'rotate_back_query_point_array']

        # Bx#pointx3 -[feature_encoder]-> BxMxC and BxMx3
        rotate_back_encode_feature, rotate_back_coarse_point_cloud = self.feature_encoder(
            rotate_back_query_point_array)

        data['predictions'][
            'rotate_back_encode_feature'] = rotate_back_encode_feature
        data['predictions'][
            'rotate_back_coarse_point_cloud'] = rotate_back_coarse_point_cloud
        return data

    def embedRotateBackPoints(self, data):
        # Bx#pointx3
        rotate_back_query_point_array = data['predictions'][
            'rotate_back_query_point_array']
        # BxMx3
        rotate_back_coarse_point_cloud = data['predictions'][
            'rotate_back_coarse_point_cloud']

        # Bx#pointx3 -[fps]-> BxMx3
        fps_rotate_back_query_point_array = fps(rotate_back_query_point_array,
                                                self.num_query)

        # BxMx3 + BxMx3 -[cat]-> Bx2Mx3
        rotate_back_coarse_points = torch.cat([
            rotate_back_coarse_point_cloud, fps_rotate_back_query_point_array
        ],
                                              dim=1).contiguous()

        data['predictions'][
            'rotate_back_coarse_points'] = rotate_back_coarse_points

        if self.training:
            data = self.lossRotateBackComplete(data)
        return data

    def lossRotateBackComplete(self, data):
        rotate_back_point_array = data['predictions'][
            'rotate_back_point_array']
        rotate_back_coarse_points = data['predictions'][
            'rotate_back_coarse_points']

        loss_rotate_back_coarse = self.loss_func(rotate_back_coarse_points,
                                                 rotate_back_point_array)

        data['losses']['loss_rotate_back_coarse'] = loss_rotate_back_coarse
        return data

    def decodeRotateBackPointsFeature(self, data):
        # BxMxC
        rotate_back_encode_feature = data['predictions'][
            'rotate_back_encode_feature']
        # BxMx3
        rotate_back_coarse_point_cloud = data['predictions'][
            'rotate_back_coarse_point_cloud']

        B, M, C = data['predictions']['rotate_back_encode_feature'].shape

        # BxMxC -[transpose]-> BxCxM -[increase_dim]-> Bx1024xM -[transpose]-> BxMx1024
        rotate_back_points_feature = self.increase_dim(
            rotate_back_encode_feature.transpose(1, 2)).transpose(1, 2)

        # BxMx1024 -[max]-> Bx1024
        rotate_back_global_points_feature = torch.max(
            rotate_back_points_feature, dim=1)[0]

        # Bx1024 -[unsqueeze]-> Bx1x1024 -[expand]-> BxMx1024
        rotate_back_replicate_global_points_feature = rotate_back_global_points_feature.unsqueeze(
            -2).expand(-1, M, -1)

        # BxMx1024 + BxMxC + BxMx3 -[cat]-> BxMx(C+1027)
        rotate_back_global_feature = torch.cat([
            rotate_back_replicate_global_points_feature,
            rotate_back_encode_feature, rotate_back_coarse_point_cloud
        ],
                                               dim=-1)

        # BxMx(C+1027) -[reshape]-> BMx(C+1027) -[reduce_map]-> BMxC
        rotate_back_reduce_global_feature = self.reduce_map(
            rotate_back_global_feature.reshape(B * M, -1))

        data['predictions'][
            'rotate_back_reduce_global_feature'] = rotate_back_reduce_global_feature
        return data

    def encodeRotateBackScale(self, data):
        # BMxC
        rotate_back_reduce_global_feature = data['predictions'][
            'rotate_back_reduce_global_feature']

        B, M, C = data['predictions']['rotate_back_encode_feature'].shape

        # BMxC -[reshape]-> BxMCx1 -[bbox_feature_encoder]-> BxCx1
        rotate_back_bbox_feature = self.bbox_feature_encoder(
            rotate_back_reduce_global_feature.reshape(B, -1, 1))

        # BxCx1 -[scale_decoder]-> Bx3x1 -[reshape]-> Bx3
        scale_inv = self.scale_decoder(rotate_back_bbox_feature).reshape(B, -1)

        data['predictions']['scale_inv'] = scale_inv

        if self.training:
            data = self.lossRotateBackScale(data)
        return data

    def lossRotateBackScale(self, data):
        scale_inv = data['predictions']['scale_inv']
        gt_scale_inv = data['inputs']['scale_inv']

        loss_scale_inv_l1 = self.l1_loss(scale_inv, gt_scale_inv)

        data['losses']['loss_scale_inv_l1'] = loss_scale_inv_l1
        return data

    @torch.no_grad()
    def scaleBackPoints(self, data):
        rotate_back_point_array = data['predictions'][
            'rotate_back_point_array']
        # Bx#pointx3
        rotate_back_query_point_array = data['predictions'][
            'rotate_back_query_point_array']
        # Bx3
        scale_inv = data['predictions']['scale_inv']

        device = rotate_back_query_point_array.device

        trans_back_points_list = []
        query_points_list = []
        trans_back_bbox_list = []
        trans_back_center_list = []

        translate = torch.tensor([0.0, 0.0, 0.0],
                                 dtype=torch.float32).to(device)
        euler_angle = torch.tensor([0.0, 0.0, 0.0],
                                   dtype=torch.float32).to(device)
        for i in range(rotate_back_query_point_array.shape[0]):
            rotate_back_points = rotate_back_point_array[i]
            rotate_back_query_points = rotate_back_query_point_array[i]
            scale = scale_inv[i]

            trans_back_points = transPointArray(rotate_back_points, translate,
                                                euler_angle, scale, True,
                                                translate)
            query_points = transPointArray(rotate_back_query_points, translate,
                                           euler_angle, scale, True, translate)

            min_point = torch.min(trans_back_points, 0)[0]
            max_point = torch.max(trans_back_points, 0)[0]

            trans_back_bbox = torch.cat([min_point, max_point])
            min_max_point = trans_back_bbox.reshape(2, 3)
            trans_back_center = torch.mean(min_max_point, 0)

            trans_back_points_list.append(trans_back_points.unsqueeze(0))
            query_points_list.append(query_points.unsqueeze(0))
            trans_back_bbox_list.append(trans_back_bbox.unsqueeze(0))
            trans_back_center_list.append(trans_back_center.unsqueeze(0))

        trans_back_point_array = torch.cat(trans_back_points_list).detach()
        query_point_array = torch.cat(query_points_list).detach()
        trans_back_bbox = torch.cat(trans_back_bbox_list).detach()
        trans_back_center = torch.cat(trans_back_center_list).detach()

        data['predictions']['trans_back_point_array'] = trans_back_point_array
        data['predictions']['query_point_array'] = query_point_array
        data['predictions']['trans_back_bbox'] = trans_back_bbox
        data['predictions']['trans_back_center'] = trans_back_center
        return data

    def encodePoints(self, data):
        # Bx#pointx3
        query_point_array = data['predictions']['query_point_array']

        # Bx#pointx3 -[feature_encoder]-> BxMxC and BxMx3
        encode_feature, coarse_point_cloud = self.feature_encoder(
            query_point_array)

        data['predictions']['encode_feature'] = encode_feature
        data['predictions']['coarse_point_cloud'] = coarse_point_cloud
        return data

    def decodePointsFeature(self, data):
        # BxMxC
        encode_feature = data['predictions']['encode_feature']
        # BxMx3
        coarse_point_cloud = data['predictions']['coarse_point_cloud']

        B, M, C = data['predictions']['encode_feature'].shape

        # BxMxC -[transpose]-> BxCxM -[increase_dim]-> Bx1024xM -[transpose]-> BxMx1024
        points_feature = self.increase_dim(encode_feature.transpose(
            1, 2)).transpose(1, 2)

        # BxMx1024 -[max]-> Bx1024
        global_points_feature = torch.max(points_feature, dim=1)[0]

        # Bx1024 -[unsqueeze]-> Bx1x1024 -[expand]-> BxMx1024
        replicate_global_points_feature = global_points_feature.unsqueeze(
            -2).expand(-1, M, -1)

        # BxMx1024 + BxMxC + BxMx3 -[cat]-> BxMx(C+1027)
        global_feature = torch.cat([
            replicate_global_points_feature, encode_feature, coarse_point_cloud
        ],
                                   dim=-1)

        # BxMx(C+1027) -[reshape]-> BMx(C+1027) -[reduce_map]-> BMxC
        reduce_global_feature = self.reduce_map(
            global_feature.reshape(B * M, -1))

        data['predictions']['reduce_global_feature'] = reduce_global_feature
        return data

    def encodeBBox(self, data):
        # BMxC
        reduce_global_feature = data['predictions']['reduce_global_feature']

        B, M, C = data['predictions']['encode_feature'].shape

        # BMxC -[reshape]-> BxMCx1 -[bbox_feature_encoder]-> BxCx1
        bbox_feature = self.bbox_feature_encoder(
            reduce_global_feature.reshape(B, -1, 1))

        # BxCx1 -[bbox_decoder]-> Bx6x1 -[reshape]-> Bx6
        bbox = self.bbox_decoder(bbox_feature).reshape(B, -1)

        # BxCx1 -[center_decoder]-> Bx3x1 -[reshape]-> Bx3
        center = self.center_decoder(bbox_feature).reshape(B, -1)

        data['predictions']['bbox'] = bbox
        data['predictions']['center'] = center

        if self.training:
            data = self.lossBBox(data)
        return data

    def lossBBox(self, data):
        bbox = data['predictions']['bbox']
        center = data['predictions']['center']
        gt_bbox = data['predictions']['trans_back_bbox']
        gt_center = data['predictions']['trans_back_center']

        loss_bbox_l1 = self.l1_loss(bbox, gt_bbox)
        loss_center_l1 = self.l1_loss(center, gt_center)
        loss_bbox_eiou = torch.mean(IoULoss.EIoU(bbox, gt_bbox))

        data['losses']['loss_bbox_l1'] = loss_bbox_l1
        data['losses']['loss_center_l1'] = loss_center_l1
        data['losses']['loss_bbox_eiou'] = loss_bbox_eiou
        return data

    def decodePatchPoints(self, data):
        # BMxC
        reduce_global_feature = data['predictions']['reduce_global_feature']
        # BxMx3
        coarse_point_cloud = data['predictions']['coarse_point_cloud']

        B, M, C = data['predictions']['encode_feature'].shape

        # BMxC -[foldingnet]-> BMx3xS -[reshape]-> BxMx3xS
        relative_patch_points = self.foldingnet(reduce_global_feature).reshape(
            B, M, 3, -1)

        # BxMx3xS + BxMx3x1 = BxMx3xS -[transpose]-> BxMxSx3
        rebuild_patch_points = (relative_patch_points +
                                coarse_point_cloud.unsqueeze(-1)).transpose(
                                    2, 3)

        # BxMxSx3 -[reshape]-> BxMSx3
        rebuild_points = rebuild_patch_points.reshape(B, -1, 3)

        data['predictions']['rebuild_patch_points'] = rebuild_patch_points
        data['predictions']['rebuild_points'] = rebuild_points
        return data

    def embedPoints(self, data):
        # Bx#pointx3
        query_point_array = data['predictions']['query_point_array']
        # BxMx3
        coarse_point_cloud = data['predictions']['coarse_point_cloud']
        # BxMSx3
        rebuild_points = data['predictions']['rebuild_points']

        # Bx#pointx3 -[fps]-> BxMx3
        fps_query_point_array = fps(query_point_array, self.num_query)

        # BxMx3 + BxMx3 -[cat]-> Bx2Mx3
        coarse_points = torch.cat([coarse_point_cloud, fps_query_point_array],
                                  dim=1).contiguous()

        # BxMSx3 + Bx#pointx3 -[cat]-> Bx(MS+#point)x3
        dense_points = torch.cat([rebuild_points, query_point_array],
                                 dim=1).contiguous()

        data['predictions']['coarse_points'] = coarse_points
        data['predictions']['dense_points'] = dense_points

        if self.training:
            data = self.lossComplete(data)
        return data

    def lossComplete(self, data):
        point_array = data['predictions']['trans_back_point_array']
        coarse_points = data['predictions']['coarse_points']
        dense_points = data['predictions']['dense_points']

        loss_coarse = self.loss_func(coarse_points, point_array)
        loss_fine = self.loss_func(dense_points, point_array)

        data['losses']['loss_coarse'] = loss_coarse
        data['losses']['loss_fine'] = loss_fine
        return data

    def cutLoss(self, data, loss_name, min_value=None, max_value=None):
        if min_value is not None:
            data['losses'][loss_name] = torch.max(
                data['losses'][loss_name],
                torch.tensor(min_value).to(torch.float32).to(
                    data['losses'][loss_name].device).reshape(1))[0]

        if max_value is not None:
            data['losses'][loss_name] = torch.min(
                data['losses'][loss_name],
                torch.tensor(max_value).to(torch.float32).to(
                    data['losses'][loss_name].device).reshape(1))[0]
        return data

    def setWeight(self,
                  data,
                  loss_name,
                  weight,
                  min_value=None,
                  max_value=None):
        if weight != 1:
            data['losses'][loss_name] = data['losses'][loss_name] * weight

        data = self.cutLoss(data, loss_name, min_value, max_value)
        return data

    def addWeight(self, data):
        if not self.training:
            return data

        self.setWeight(data, 'loss_origin_euler_angle_inv', 1)
        self.setWeight(data, 'loss_origin_query_euler_angle_inv', 1)
        self.setWeight(data, 'loss_partial_complete_euler_angle_inv_diff', 1)

        self.setWeight(data, 'loss_decode_origin_udf', 1)
        self.setWeight(data, 'loss_decode_origin_query_udf', 1)

        self.setWeight(data, 'loss_rotate_back_coarse', 1000)

        self.setWeight(data, 'loss_scale_inv_l1', 1)

        self.setWeight(data, 'loss_bbox_l1', 1000)
        self.setWeight(data, 'loss_center_l1', 1000)
        self.setWeight(data, 'loss_bbox_eiou', 10, max_value=10)

        self.setWeight(data, 'loss_coarse', 1000)
        self.setWeight(data, 'loss_fine', 1000)
        return data

    def forward(self, data):
        data = self.moveToOrigin(data)

        data = self.rotate_net(data)

        data = self.rotateBackPoints(data)

        data = self.encodeRotateBackPoints(data)

        data = self.embedRotateBackPoints(data)

        data = self.decodeRotateBackPointsFeature(data)

        data = self.encodeRotateBackScale(data)

        data = self.scaleBackPoints(data)

        data = self.encodePoints(data)

        data = self.decodePointsFeature(data)

        data = self.encodeBBox(data)

        data = self.decodePatchPoints(data)

        data = self.embedPoints(data)

        data = self.addWeight(data)
        return data
