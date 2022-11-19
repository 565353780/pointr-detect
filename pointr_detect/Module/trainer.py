#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pointr_detect.Model.pointr import PoinTr

from pointr_detect.Data.average_meter import AverageMeter

from pointr_detect.Dataset.shapenet_55 import ShapeNet55Dataset

from pointr_detect.Scheduler.bn_momentum import BNMomentumScheduler

from pointr_detect.Method.time import getCurrentTime
from pointr_detect.Method.sample import seprate_point_cloud
from pointr_detect.Method.move import moveToOrigin, moveToMeanPoint
from pointr_detect.Method.device import toCuda
from pointr_detect.Method.path import createFileFolder, renameFile, removeFile
from pointr_detect.Method.render import renderPointArrayWithUnitBBox


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class Trainer(object):

    def __init__(self):
        self.model = PoinTr().cuda()

        self.dataset = ShapeNet55Dataset()
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=16,
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=24,
                                     worker_init_fn=worker_init_fn)

        self.lr = 5e-4
        self.weight_decay = 5e-4
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
        self.epoch = 0

        self.optimizer = AdamW(self.model.parameters(),
                               lr=self.lr,
                               weight_decay=self.weight_decay)
        lr_lambda = lambda e: max(self.lr_decay**
                                  (e / self.decay_step), self.lowest_decay)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        bnm_lambda = lambda e: max(
            self.bn_momentum * self.bn_decay**
            (e / self.bn_decay_step), self.bn_lowest_decay)
        self.bnscheduler = BNMomentumScheduler(self.model, bnm_lambda)
        self.summary_writer = None
        return

    def loadSummaryWriter(self):
        self.summary_writer = SummaryWriter("./logs/" + self.log_folder_name +
                                            "/")
        return True

    def loadModel(self, model_file_path):
        if not os.path.exists(model_file_path):
            self.loadSummaryWriter()
            print("[WARN][Trainer::loadModel]")
            print("\t model_file not exist! start training from step 0...")
            return True

        model_dict = torch.load(model_file_path)

        self.model.load_state_dict(model_dict['pointr_model'])
        self.optimizer.load_state_dict(model_dict['optimizer'])
        self.step = model_dict['step']
        self.loss_min = model_dict['loss_min']
        self.log_folder_name = model_dict['log_folder_name']
        self.epoch = model_dict['epoch']

        self.loadSummaryWriter()
        print("[INFO][Trainer::loadModel]")
        print("\t load model success! start training from step " +
              str(self.step) + "...")
        return True

    def saveModel(self, save_model_file_path):
        model_dict = {
            'pointr_model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'loss_min': self.loss_min,
            'log_folder_name': self.log_folder_name,
            'epoch': self.epoch
        }

        createFileFolder(save_model_file_path)

        tmp_save_model_file_path = save_model_file_path.split(
            ".pth")[0] + "_tmp.pth"

        torch.save(model_dict, tmp_save_model_file_path)

        removeFile(save_model_file_path)
        renameFile(tmp_save_model_file_path, save_model_file_path)
        return True

    def testTrain(self):
        for data in tqdm(self.dataloader):
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

    def trainStep(self, data):
        toCuda(data)

        sample_point_array, _ = seprate_point_cloud(
            data['inputs']['point_array'], [0.25, 0.75])

        #  points = sample_point_array.cpu().numpy()[0]
        #  points = moveToOrigin(points).reshape(1, -1, 3)
        #  sample_point_array = torch.tensor(points).cuda()

        data['inputs']['sample_point_array'] = sample_point_array

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

        while self.epoch < total_epoch:
            print("[INFO][Trainer::train]")
            print("\t start training, epoch : " + str(self.epoch + 1) + "/" +
                  str(total_epoch) + "...")

            for_data = self.dataloader
            if print_progress:
                for_data = tqdm(for_data)
            for data in for_data:
                self.trainStep(data)
                self.step += 1

            self.epoch += 1

            self.scheduler.step(self.epoch)
            self.bn_scheduler.step(self.epoch)

            self.saveModel("./output/" + self.log_folder_name +
                           "/model_last.pth")
        return True
