#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import open3d as o3d
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from points_shape_detect.Dataset.cad_dataset import CADDataset
from points_shape_detect.Method.device import toCuda
from points_shape_detect.Method.path import (createFileFolder, removeFile,
                                             renameFile)
from points_shape_detect.Method.render import (renderPointArray,
                                               renderPointArrayList,
                                               renderRotateBackPoints,
                                               renderPredictBBox,
                                               renderTransBackPoints)
from points_shape_detect.Method.sample import seprate_point_cloud
from points_shape_detect.Method.time import getCurrentTime
from points_shape_detect.Method.trans import getInverseTrans, transPointArray
from points_shape_detect.Scheduler.bn_momentum import BNMomentumScheduler

from points_shape_detect.Model.rotate_net import RotateNet
from points_shape_detect.Model.bbox_net import BBoxNet
from points_shape_detect.Model.points_shape_net import PointsShapeNet


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class Trainer(object):

    def __init__(self):
        self.batch_size = 24
        self.lr = 5e-6
        self.weight_decay = 5e-6
        self.decay_step = 21
        self.lr_decay = 0.76
        self.lowest_decay = 0.02
        self.bn_decay_step = 21
        self.bn_decay = 0.5
        self.bn_momentum = 0.9
        self.bn_lowest_decay = 0.01
        self.step = 0
        self.loss_min = float('inf')
        self.log_folder_name = getCurrentTime()

        self.model = BBoxNet().cuda()
        #  self.model = RotateNet().cuda()
        #  self.model = PointsShapeNet().cuda()

        self.dataset = CADDataset()
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=self.batch_size,
                                     worker_init_fn=worker_init_fn)

        self.optimizer = AdamW(self.model.parameters(),
                               lr=self.lr,
                               weight_decay=self.weight_decay)
        lr_lambda = lambda e: max(self.lr_decay**
                                  (e / self.decay_step), self.lowest_decay)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        bnm_lambda = lambda e: max(
            self.bn_momentum * self.bn_decay**
            (e / self.bn_decay_step), self.bn_lowest_decay)
        self.bn_scheduler = BNMomentumScheduler(self.model, bnm_lambda)
        self.summary_writer = None
        return

    def loadSummaryWriter(self):
        self.summary_writer = SummaryWriter("./logs/" + self.log_folder_name +
                                            "/")
        return True

    def loadModel(self, model_file_path, resume_model_only=False):
        if not os.path.exists(model_file_path):
            self.loadSummaryWriter()
            print("[WARN][Trainer::loadModel]")
            print("\t model_file not exist! start training from step 0...")
            return True

        model_dict = torch.load(model_file_path)

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in model_dict['model'].items():
            if 'euler_angle_decoder' in k:
                continue
            new_state_dict[k] = v

        #  self.model.load_state_dict(model_dict['model'])
        self.model.load_state_dict(new_state_dict)

        if not resume_model_only:
            self.optimizer.load_state_dict(model_dict['optimizer'])
            self.step = model_dict['step']
            self.loss_min = model_dict['loss_min']
            self.log_folder_name = model_dict['log_folder_name']

        self.loadSummaryWriter()
        print("[INFO][Trainer::loadModel]")
        print("\t load model success! start training from step " +
              str(self.step) + "...")
        return True

    def saveModel(self, save_model_file_path):
        model_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'loss_min': self.loss_min,
            'log_folder_name': self.log_folder_name,
        }

        createFileFolder(save_model_file_path)

        tmp_save_model_file_path = save_model_file_path.split(
            ".pth")[0] + "_tmp.pth"

        torch.save(model_dict, tmp_save_model_file_path)

        removeFile(save_model_file_path)
        renameFile(tmp_save_model_file_path, save_model_file_path)
        return True

    @torch.no_grad()
    def preProcessData(self, data):
        trans_point_array = data['inputs']['trans_point_array']

        trans_query_point_array, _ = seprate_point_cloud(
            trans_point_array, [0.0, 0.75])

        data['inputs']['trans_query_point_array'] = trans_query_point_array
        return data

    def testTrain(self):
        test_dataloader = DataLoader(self.dataset,
                                     batch_size=2,
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=1,
                                     worker_init_fn=worker_init_fn)

        for data in tqdm(test_dataloader):
            toCuda(data)
            data = self.preProcessData(data)

            renderPointArrayList([
                data['inputs']['trans_query_point_array'][0],
                data['inputs']['trans_point_array'][0],
            ])

            data = self.model(data)

            print(data['predictions'].keys())
            renderRotateBackPoints(data)
            renderTransBackPoints(data)
            #  renderPredictBBox(data)
        return True

    def trainStep(self, data):
        toCuda(data)
        data = self.preProcessData(data)

        self.model.train()
        self.model.zero_grad()
        self.optimizer.zero_grad()

        data = self.model(data)

        losses = data['losses']

        losses_tensor = torch.cat([
            loss if len(loss.shape) > 0 else loss.reshape(1)
            for loss in data['losses'].values()
        ])

        loss_sum = torch.sum(losses_tensor)
        loss_sum_float = loss_sum.detach().cpu().numpy()
        self.summary_writer.add_scalar("Loss/loss_sum", loss_sum_float,
                                       self.step)

        if loss_sum_float < self.loss_min:
            self.loss_min = loss_sum_float
            self.saveModel("./output/" + self.log_folder_name +
                           "/model_best.pth")

        for key, loss in losses.items():
            loss_tensor = loss.detach() if len(
                loss.shape) > 0 else loss.detach().reshape(1)
            loss_mean = torch.mean(loss_tensor)
            self.summary_writer.add_scalar("Loss/" + key, loss_mean, self.step)

        loss_sum.backward()
        self.optimizer.step()
        return True

    def train(self, print_progress=False):
        total_epoch = 10000000

        for epoch in range(total_epoch):
            print("[INFO][Trainer::train]")
            print("\t start training, epoch : " + str(epoch + 1) + "/" +
                  str(total_epoch) + "...")

            self.summary_writer.add_scalar(
                "Lr/lr",
                self.optimizer.state_dict()['param_groups'][0]['lr'],
                self.step)

            for_data = self.dataloader
            if print_progress:
                for_data = tqdm(for_data)
            for data in for_data:
                self.trainStep(data)
                self.step += 1

            self.scheduler.step()
            self.bn_scheduler.step()

            self.saveModel("./output/" + self.log_folder_name +
                           "/model_last.pth")
        return True
