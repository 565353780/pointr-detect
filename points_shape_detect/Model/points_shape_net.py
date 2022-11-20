#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from pointnet2_ops import pointnet2_utils
from torch import nn

from points_shape_detect.Lib.chamfer_dist import ChamferDistanceL1
from points_shape_detect.Method.sample import fps
from points_shape_detect.Model.fold import Fold
from points_shape_detect.Model.pc_transformer import PCTransformer

from points_shape_detect.Loss.ious import IoULoss


class PointsShapeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.trans_dim = 384
        self.knn_layer = 1
        self.num_pred = 6144
        self.num_query = 96

        self.fold_step = int(pow(self.num_pred // self.num_query, 0.5) + 0.5)

        self.base_model = PCTransformer(in_chans=3,
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

        self.bbox_feature_decoder = nn.Sequential(
            nn.Conv1d(self.num_query * self.trans_dim, self.trans_dim, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, self.trans_dim, 1))

        self.bbox_decoder = nn.Sequential(nn.Conv1d(self.trans_dim, 6, 1),
                                          nn.LeakyReLU(negative_slope=0.2),
                                          nn.Conv1d(6, 6, 1))

        self.center_decoder = nn.Sequential(nn.Conv1d(self.trans_dim, 3, 1),
                                            nn.LeakyReLU(negative_slope=0.2),
                                            nn.Conv1d(3, 3, 1))

        self.loss_func = ChamferDistanceL1()
        self.l1_loss = nn.SmoothL1Loss()
        return

    def get_loss(self, data):
        loss_coarse = self.loss_func(data['predictions']['coarse_points'],
                                     data['inputs']['point_array'])
        loss_fine = self.loss_func(data['predictions']['dense_points'],
                                   data['inputs']['point_array'])

        loss_bbox_l1 = self.l1_loss(data['predictions']['bbox'],
                                    data['inputs']['bbox'])
        loss_center_l1 = self.l1_loss(data['predictions']['center'],
                                      data['inputs']['center'])

        loss_bbox_eiou = torch.mean(
            IoULoss.EIoU(data['predictions']['bbox'], data['inputs']['bbox']))

        data['losses']['loss_coarse'] = loss_coarse * 1000
        data['losses']['loss_fine'] = loss_fine * 1000
        data['losses']['loss_bbox_l1'] = loss_bbox_l1 * 1000
        data['losses']['loss_center_l1'] = loss_center_l1 * 1000
        data['losses']['loss_bbox_eiou'] = loss_bbox_eiou * 10
        return data

    def forward(self, data):
        # Bx#pointx3
        point_array = data['inputs']['point_array']

        # Bx#pointx3 -[base_model]-> BxMxC and BxMx3
        q, coarse_point_cloud = self.base_model(point_array)

        B, M, C = q.shape

        # BxMxC -[transpose]-> BxCxM -[increase_dim]-> Bx1024xM -[transpose]-> BxMx1024
        points_feature = self.increase_dim(q.transpose(1, 2)).transpose(1, 2)

        # BxMx1024 -[max]-> Bx1024
        global_points_feature = torch.max(points_feature, dim=1)[0]

        # Bx1024 -[unsqueeze]-> Bx1x1024 -[expand]-> BxMx1024
        replicate_global_points_feature = global_points_feature.unsqueeze(
            -2).expand(-1, M, -1)

        # BxMx1024 + BxMxC + BxMx3 -[cat]-> BxMx(C+1027)
        global_feature = torch.cat(
            [replicate_global_points_feature, q, coarse_point_cloud], dim=-1)

        # BxMx(C+1027) -[reshape]-> BMx(C+1027) -[reduce_map]-> BMxC
        reduce_global_feature = self.reduce_map(
            global_feature.reshape(B * M, -1))

        # BMxC -[reshape]-> BxMCx1 -[bbox_feature_decoder]-> BxCx1
        bbox_feature = self.bbox_feature_decoder(
            reduce_global_feature.reshape(B, -1, 1))

        # BxCx1 -[bbox_decoder]-> Bx6x1 -[reshape]-> Bx6
        bbox = self.bbox_decoder(bbox_feature).reshape(B, -1)

        # BxCx1 -[center_decoder]-> Bx3x1 -[reshape]-> Bx3
        center = self.center_decoder(bbox_feature).reshape(B, -1)

        # BMxC -[foldingnet]-> BMx3xS -[reshape]-> BxMx3xS
        relative_patch_points = self.foldingnet(reduce_global_feature).reshape(
            B, M, 3, -1)

        # BxMx3xS + BxMx3x1 = BxMx3xS -[transpose]-> BxMxSx3
        rebuild_patch_points = (relative_patch_points +
                                coarse_point_cloud.unsqueeze(-1)).transpose(
                                    2, 3)

        # BxMxSx3 -[reshape]-> BxMSx3
        rebuild_points = rebuild_patch_points.reshape(B, -1, 3)

        # Bx#pointx3 -[fps]-> BxMx3
        sample_point_array = fps(point_array, self.num_query)

        # BxMx3 + BxMx3 -[cat]-> Bx2Mx3
        coarse_points = torch.cat([coarse_point_cloud, sample_point_array],
                                  dim=1).contiguous()

        # BxMSx3 + Bx#pointx3 -[cat]-> Bx(MS+#point)x3
        dense_points = torch.cat([rebuild_points, point_array],
                                 dim=1).contiguous()

        data['predictions']['coarse_point_cloud'] = coarse_point_cloud
        data['predictions']['rebuild_patch_points'] = rebuild_patch_points
        data['predictions']['rebuild_points'] = rebuild_points
        data['predictions']['coarse_points'] = coarse_points
        data['predictions']['dense_points'] = dense_points
        data['predictions']['bbox'] = bbox
        data['predictions']['center'] = center

        if self.training:
            data = self.get_loss(data)
        return data
