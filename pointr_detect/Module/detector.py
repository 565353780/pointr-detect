#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm

from pointr_detect.Model.pointr import PoinTr

from pointr_detect.Method.sample import fps, seprate_point_cloud
from pointr_detect.Method.move import moveToOrigin, moveToMeanPoint
from pointr_detect.Method.device import toCuda
from pointr_detect.Method.render import renderPointArrayWithUnitBBox


class Detector(object):

    def __init__(self, model_file_path=None):
        self.model = PoinTr().cuda()

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadPoinTrModel(self, model_file_path):
        assert os.path.exists(model_file_path)

        print("[INFO][Detector::loadModel]")
        print("\t start loading pointr model from :")
        print("\t", model_file_path)

        state_dict = torch.load(model_file_path)
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

    def loadModel(self, model_file_path):
        assert os.path.exists(model_file_path)

        print("[INFO][Detector::loadModel]")
        print("\t start loading model from :")
        print("\t", model_file_path)
        model_dict = torch.load(model_file_path)
        self.model.load_state_dict(model_dict['pointr_model'])
        return True

    def detectPointArray(self, point_array):
        self.model.eval()

        data = {'inputs': {}, 'predictions': {}, 'losses': {}, 'logs': {}}

        data['inputs']['point_array'] = torch.from_numpy(
            point_array.reshape(1, -1, 3).astype(np.float32)).cuda()

        data['inputs']['sample_point_array'] = fps(
            data['inputs']['point_array'], 2048)

        data = self.model(data)
        return data
