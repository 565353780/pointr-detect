#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm

from points_shape_detect.Model.points_shape_net import PointsShapeNet

from points_shape_detect.Method.sample import fps
from points_shape_detect.Method.device import toCuda


class Detector(object):

    def __init__(self, model_file_path=None):
        self.model = PointsShapeNet().cuda()

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(self, model_file_path):
        assert os.path.exists(model_file_path)

        print("[INFO][Detector::loadModel]")
        print("\t start loading model from :")
        print("\t", model_file_path)
        model_dict = torch.load(model_file_path)
        self.model.load_state_dict(model_dict['model'])
        return True

    def detectPointArray(self, point_array):
        self.model.eval()

        data = {'inputs': {}, 'predictions': {}, 'losses': {}, 'logs': {}}

        data['inputs']['point_array'] = torch.from_numpy(
            point_array.reshape(1, -1, 3).astype(np.float32)).cuda()

        data['inputs']['query_point_array'] = fps(
            data['inputs']['point_array'], 2048)

        data = self.model(data)
        return data
