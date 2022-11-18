#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from pointr_detect.Model.pointr import PoinTr

from pointr_detect.Data.average_meter import AverageMeter

from pointr_detect.Dataset.shapenet_55 import ShapeNet55Dataset

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class Detector(object):

    def __init__(self, model_file_path=None):
        self.model_file_path = None

        self.model = PoinTr().cuda()

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(self, model_file_path):
        self.model_file_path = model_file_path

        if os.path.exists(self.model_file_path):
            print("[INFO][Detector::loadModel]")
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

    def detectPointArray(self, point_array):
        self.model.eval()

        data = {'inputs': {}, 'predictions': {}, 'losses': {}, 'logs': {}}

        dataset = ShapeNet55Dataset()
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            worker_init_fn=worker_init_fn)
        return data
