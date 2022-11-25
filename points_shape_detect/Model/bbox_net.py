#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn

from points_shape_detect.Lib.chamfer_dist import ChamferDistanceL1
from points_shape_detect.Loss.ious import IoULoss
from points_shape_detect.Method.sample import fps
from points_shape_detect.Method.weight import setWeight
from points_shape_detect.Model.fold import Fold
from points_shape_detect.Model.pc_transformer import PCTransformer


class BBoxNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.trans_dim = 384
        self.knn_layer = 1
        self.num_pred = 6144
        self.num_query = 96

        self.fold_step = int(pow(self.num_pred // self.num_query, 0.5) + 0.5)

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
        origin_bbox_list = []
        origin_center_list = []

        for i in range(trans_query_point_array.shape[0]):
            trans_points = trans_point_array[i]
            trans_query_points = trans_query_point_array[i]
            trans_query_points_center = torch.mean(trans_query_points, 0)

            origin_points = trans_points - trans_query_points_center
            origin_query_points = trans_query_points - trans_query_points_center

            min_point = torch.min(origin_points, 0)[0]
            max_point = torch.max(origin_points, 0)[0]

            origin_bbox = torch.cat([min_point, max_point])
            min_max_point = origin_bbox.reshape(2, 3)
            origin_center = torch.mean(min_max_point, 0)

            origin_points_list.append(origin_points.unsqueeze(0))
            origin_query_points_list.append(origin_query_points.unsqueeze(0))
            origin_bbox_list.append(origin_bbox.unsqueeze(0))
            origin_center_list.append(origin_center.unsqueeze(0))

        origin_point_array = torch.cat(origin_points_list).detach()
        origin_query_point_array = torch.cat(origin_query_points_list).detach()
        origin_bbox = torch.cat(origin_bbox_list).detach()
        origin_center = torch.cat(origin_center_list).detach()

        data['inputs']['origin_point_array'] = origin_point_array
        data['inputs']['origin_query_point_array'] = origin_query_point_array
        data['inputs']['origin_bbox'] = origin_bbox
        data['inputs']['origin_center'] = origin_center
        return data

    def encodeOriginPoints(self, data):
        # Bx#pointx3
        origin_query_point_array = data['inputs']['origin_query_point_array']

        # Bx#pointx3 -[feature_encoder]-> BxMxC and BxMx3
        origin_encode_feature, origin_coarse_point_cloud = self.feature_encoder(
            origin_query_point_array)

        data['predictions']['origin_encode_feature'] = origin_encode_feature
        data['predictions'][
            'origin_coarse_point_cloud'] = origin_coarse_point_cloud
        return data

    def decodeOriginPointsFeature(self, data):
        # BxMxC
        origin_encode_feature = data['predictions']['origin_encode_feature']
        # BxMx3
        origin_coarse_point_cloud = data['predictions'][
            'origin_coarse_point_cloud']

        B, M, C = data['predictions']['origin_encode_feature'].shape

        # BxMxC -[transpose]-> BxCxM -[increase_dim]-> Bx1024xM -[transpose]-> BxMx1024
        origin_points_feature = self.increase_dim(
            origin_encode_feature.transpose(1, 2)).transpose(1, 2)

        # BxMx1024 -[max]-> Bx1024
        origin_global_points_feature = torch.max(origin_points_feature,
                                                 dim=1)[0]

        # Bx1024 -[unsqueeze]-> Bx1x1024 -[expand]-> BxMx1024
        origin_replicate_global_points_feature = origin_global_points_feature.unsqueeze(
            -2).expand(-1, M, -1)

        # BxMx1024 + BxMxC + BxMx3 -[cat]-> BxMx(C+1027)
        origin_global_feature = torch.cat([
            origin_replicate_global_points_feature, origin_encode_feature,
            origin_coarse_point_cloud
        ],
                                          dim=-1)

        # BxMx(C+1027) -[reshape]-> BMx(C+1027) -[reduce_map]-> BMxC
        origin_reduce_global_feature = self.reduce_map(
            origin_global_feature.reshape(B * M, -1))

        data['predictions'][
            'origin_reduce_global_feature'] = origin_reduce_global_feature
        return data

    def encodeScale(self, data):
        # BMxC
        origin_reduce_global_feature = data['predictions'][
            'origin_reduce_global_feature']

        B, M, C = data['predictions']['origin_encode_feature'].shape

        # BMxC -[reshape]-> BxMCx1 -[bbox_feature_encoder]-> BxCx1
        origin_bbox_feature = self.bbox_feature_encoder(
            origin_reduce_global_feature.reshape(B, -1, 1))

        # BxCx1 -[scale_decoder]-> Bx3x1 -[reshape]-> Bx3
        scale_inv = self.scale_decoder(origin_bbox_feature).reshape(B, -1)

        data['predictions']['scale_inv'] = scale_inv

        #  if self.training:
        data = self.lossScale(data)
        return data

    def lossScale(self, data):
        scale_inv = data['predictions']['scale_inv']
        gt_scale_inv = data['inputs']['scale_inv']

        loss_scale_inv_l1 = self.l1_loss(scale_inv, gt_scale_inv)

        data['losses']['loss_scale_inv_l1'] = loss_scale_inv_l1
        return data

    def encodeOriginBBox(self, data):
        # BMxC
        origin_reduce_global_feature = data['predictions'][
            'origin_reduce_global_feature']

        B, M, C = data['predictions']['origin_encode_feature'].shape

        # BMxC -[reshape]-> BxMCx1 -[bbox_feature_encoder]-> BxCx1
        origin_bbox_feature = self.bbox_feature_encoder(
            origin_reduce_global_feature.reshape(B, -1, 1))

        # BxCx1 -[bbox_decoder]-> Bx6x1 -[reshape]-> Bx6
        origin_bbox = self.bbox_decoder(origin_bbox_feature).reshape(B, -1)

        # BxCx1 -[center_decoder]-> Bx3x1 -[reshape]-> Bx3
        origin_center = self.center_decoder(origin_bbox_feature).reshape(B, -1)

        data['predictions']['origin_bbox'] = origin_bbox
        data['predictions']['origin_center'] = origin_center

        #  if self.training:
        data = self.lossOriginBBox(data)
        return data

    def lossOriginBBox(self, data):
        origin_bbox = data['predictions']['origin_bbox']
        origin_center = data['predictions']['origin_center']
        gt_origin_bbox = data['inputs']['origin_bbox']
        gt_origin_center = data['inputs']['origin_center']

        loss_origin_bbox_l1 = self.l1_loss(origin_bbox, gt_origin_bbox)
        loss_origin_center_l1 = self.l1_loss(origin_center, gt_origin_center)
        loss_origin_bbox_eiou = torch.mean(
            IoULoss.EIoU(origin_bbox, gt_origin_bbox))

        data['losses']['loss_origin_bbox_l1'] = loss_origin_bbox_l1
        data['losses']['loss_origin_center_l1'] = loss_origin_center_l1
        data['losses']['loss_origin_bbox_eiou'] = loss_origin_bbox_eiou
        return data

    def decodeOriginPatchPoints(self, data):
        # BMxC
        origin_reduce_global_feature = data['predictions'][
            'origin_reduce_global_feature']
        # BxMx3
        origin_coarse_point_cloud = data['predictions'][
            'origin_coarse_point_cloud']

        B, M, C = data['predictions']['origin_encode_feature'].shape

        # BMxC -[foldingnet]-> BMx3xS -[reshape]-> BxMx3xS
        origin_relative_patch_points = self.foldingnet(
            origin_reduce_global_feature).reshape(B, M, 3, -1)

        # BxMx3xS + BxMx3x1 = BxMx3xS -[transpose]-> BxMxSx3
        origin_rebuild_patch_points = (
            origin_relative_patch_points +
            origin_coarse_point_cloud.unsqueeze(-1)).transpose(2, 3)

        # BxMxSx3 -[reshape]-> BxMSx3
        origin_rebuild_points = origin_rebuild_patch_points.reshape(B, -1, 3)

        data['predictions'][
            'origin_rebuild_patch_points'] = origin_rebuild_patch_points
        data['predictions']['origin_rebuild_points'] = origin_rebuild_points
        return data

    def embedOriginPoints(self, data):
        # Bx#pointx3
        origin_query_point_array = data['inputs']['origin_query_point_array']
        # BxMx3
        origin_coarse_point_cloud = data['predictions'][
            'origin_coarse_point_cloud']
        # BxMSx3
        origin_rebuild_points = data['predictions']['origin_rebuild_points']

        # Bx#pointx3 -[fps]-> BxMx3
        fps_origin_query_point_array = fps(origin_query_point_array,
                                           self.num_query)

        # BxMx3 + BxMx3 -[cat]-> Bx2Mx3
        origin_coarse_points = torch.cat(
            [origin_coarse_point_cloud, fps_origin_query_point_array],
            dim=1).contiguous()

        # BxMSx3 + Bx#pointx3 -[cat]-> Bx(MS+#point)x3
        origin_dense_points = torch.cat(
            [origin_rebuild_points, origin_query_point_array],
            dim=1).contiguous()

        data['predictions']['origin_coarse_points'] = origin_coarse_points
        data['predictions']['origin_dense_points'] = origin_dense_points

        #  if self.training:
        data = self.lossOriginComplete(data)
        return data

    def lossOriginComplete(self, data):
        origin_point_array = data['inputs']['origin_point_array']
        origin_coarse_points = data['predictions']['origin_coarse_points']
        origin_dense_points = data['predictions']['origin_dense_points']

        loss_origin_coarse = self.loss_func(origin_coarse_points,
                                            origin_point_array)
        loss_origin_fine = self.loss_func(origin_dense_points,
                                          origin_point_array)

        data['losses']['loss_origin_coarse'] = loss_origin_coarse
        data['losses']['loss_origin_fine'] = loss_origin_fine
        return data

    def addWeight(self, data):
        #  if not self.training:
            #  return data

        data = setWeight(data, 'loss_scale_inv_l1', 1000)

        data = setWeight(data, 'loss_origin_bbox_l1', 1000)
        data = setWeight(data, 'loss_origin_center_l1', 1000)
        data = setWeight(data, 'loss_origin_bbox_eiou', 100, max_value=100)

        data = setWeight(data, 'loss_origin_coarse', 1000)
        data = setWeight(data, 'loss_origin_fine', 1000)
        return data

    def forward(self, data):
        data = self.moveToOrigin(data)

        data = self.encodeOriginPoints(data)

        data = self.decodeOriginPointsFeature(data)

        data = self.encodeScale(data)

        data = self.encodeOriginBBox(data)

        data = self.decodeOriginPatchPoints(data)

        data = self.embedOriginPoints(data)

        data = self.addWeight(data)
        return data
