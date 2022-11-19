#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm

from pointr_detect.Model.pointr import PoinTr

from pointr_detect.Data.average_meter import AverageMeter

from pointr_detect.Dataset.shapenet_55 import ShapeNet55Dataset

from pointr_detect.Method.sample import seprate_point_cloud
from pointr_detect.Method.move import moveToOrigin, moveToMeanPoint
from pointr_detect.Method.device import toCuda
from pointr_detect.Method.render import renderPointArrayWithUnitBBox


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class Trainer(object):

    def __init__(self, model_file_path=None):
        self.model_file_path = None

        self.model = PoinTr().cuda()

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(self, model_file_path):
        self.model_file_path = model_file_path

        if os.path.exists(self.model_file_path):
            print("[INFO][Trainer::loadModel]")
            print("\t start loading model from :")
            print("\t", self.model_file_path)

            state_dict = torch.load(self.model_file_path)
            if state_dict.get('model') is not None:
                base_ckpt = {
                    k.replace("module.", ""): v
                    for k, v in state_dict['model'].items()
                }
            elif state_dict.get('base_model') is not None:
                base_ckpt = {
                    k.replace("module.", ""): v
                    for k, v in state_dict['base_model'].items()
                }
            self.model.load_state_dict(base_ckpt)
        return True

    def testTrain(self):
        dataset = ShapeNet55Dataset()
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 drop_last=False,
                                                 num_workers=4,
                                                 worker_init_fn=worker_init_fn)
        for data in tqdm(dataloader):
            toCuda(data)

            sample_point_array, _ = seprate_point_cloud(
                data['inputs']['point_array'], [0.25, 0.75])

            #  points = sample_point_array.cpu().numpy()[0]
            #  points = moveToOrigin(points).reshape(1, -1, 3)
            #  sample_point_array = torch.tensor(points).cuda()

            data['inputs']['sample_point_array'] = sample_point_array

            renderPointArrayWithUnitBBox(data['inputs']['point_array'][0])
            renderPointArrayWithUnitBBox(
                data['inputs']['sample_point_array'][0])

            data = self.model(data)

            dense_points = data['predictions']['dense_points']
            #  renderPointArrayWithUnitBBox(dense_points[0])
        return True

    def train(self):
        return True
