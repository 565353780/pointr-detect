#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from points_shape_detect.Data.io import IO
from points_shape_detect.Method.trans import (getInverseTrans,
                                              normalizePointArray,
                                              transPointArray)

sys.path.append("../auto-cad-recon")
sys.path.append("../mesh-manage/")
sys.path.append("../udf-generate/")
sys.path.append("../scannet-dataset-manage")
sys.path.append("../scan2cad-dataset-manage")
sys.path.append("../shapenet-dataset-manage")

from auto_cad_recon.Module.dataset_manager import DatasetManager


class CADDataset(Dataset):

    def __init__(self, training=True, training_percent=0.8):
        self.training = training
        self.training_percent = training_percent

        self.cad_model_file_path_list = []
        self.train_idx_list = []
        self.eval_idx_list = []

        self.loadShapeNet55()
        self.updateIdx()
        return

    def reset(self):
        self.cad_model_file_path_list = []
        return True

    def updateIdx(self):
        model_num = len(self.cad_model_file_path_list)
        if model_num == 1:
            self.train_idx_list = [0]
            self.eval_idx_list = [0]
            return True

        assert model_num > 0

        train_model_num = int(model_num * self.training_percent)
        if train_model_num == 0:
            train_model_num += 1
        elif train_model_num == model_num:
            train_model_num -= 1

        random_idx_list = np.random.choice(np.arange(model_num),
                                           size=model_num,
                                           replace=False)

        self.train_idx_list = random_idx_list[:train_model_num]
        self.eval_idx_list = random_idx_list[train_model_num:]
        return True

    def loadScan2CAD(self):
        scannet_dataset_folder_path = "/home/chli/chLi/ScanNet/scans/"
        scannet_object_dataset_folder_path = "/home/chli/chLi/ScanNet/objects/"
        scannet_bbox_dataset_folder_path = "/home/chli/chLi/ScanNet/bboxes/"
        scan2cad_dataset_folder_path = "/home/chli/chLi/Scan2CAD/scan2cad_dataset/"
        scan2cad_object_model_map_dataset_folder_path = "/home/chli/chLi/Scan2CAD/object_model_maps/"
        shapenet_dataset_folder_path = "/home/chli/chLi/ShapeNet/Core/ShapeNetCore.v2/"
        shapenet_udf_dataset_folder_path = "/home/chli/chLi/ShapeNet/udfs/"
        print_progress = True

        dataset_manager = DatasetManager(
            scannet_dataset_folder_path, scannet_object_dataset_folder_path,
            scannet_bbox_dataset_folder_path, scan2cad_dataset_folder_path,
            scan2cad_object_model_map_dataset_folder_path,
            shapenet_dataset_folder_path, shapenet_udf_dataset_folder_path)

        scene_name_list = dataset_manager.getScanNetSceneNameList()

        print("[INFO][CADDataset::loadScan2CAD]")
        print("\t start load scan2cad dataset...")
        for scene_name in tqdm(scene_name_list):
            object_file_name_list = dataset_manager.getScanNetObjectFileNameList(
                scene_name)
            for object_file_name in object_file_name_list:
                shapenet_model_dict = dataset_manager.getShapeNetModelDict(
                    scene_name, object_file_name)
                cad_model_file_path = shapenet_model_dict[
                    'shapenet_model_file_path']
                self.cad_model_file_path_list.append(cad_model_file_path)
        return True

    def loadShapeNet55(self):
        dataset_folder_path = '/home/chli/chLi/PoinTr/ShapeNet55/shapenet_pc/'
        file_name_list = os.listdir(dataset_folder_path)
        for file_name in file_name_list:
            self.cad_model_file_path_list.append(dataset_folder_path +
                                                 file_name)
        return True

    def __getitem__(self, idx, training=True):
        if self.training:
            idx = self.train_idx_list[idx]
        else:
            idx = self.eval_idx_list[idx]

        cad_model_file_path = self.cad_model_file_path_list[idx]

        data = {'inputs': {}, 'predictions': {}, 'losses': {}, 'logs': {}}

        point_array = IO.get(cad_model_file_path).astype(np.float32)
        origin_point_array = normalizePointArray(point_array)

        translate = (np.random.rand(3) - 0.5) * 1000
        euler_angle = (np.random.rand(3) - 0.5) * 360.0
        scale = 1.0 + ((np.random.rand(3) - 0.5) * 0.2)

        trans_point_array = transPointArray(origin_point_array, translate,
                                            euler_angle, scale)
        data['inputs']['trans_point_array'] = torch.from_numpy(
            trans_point_array).float()

        if training:
            translate_inv, euler_angle_inv, scale_inv = getInverseTrans(
                translate, euler_angle, scale)

            data['inputs']['euler_angle_inv'] = torch.from_numpy(
                euler_angle_inv).to(torch.float32)
            data['inputs']['scale_inv'] = torch.from_numpy(scale_inv).to(
                torch.float32)
        return data

    def __len__(self):
        if self.training:
            return len(self.train_idx_list)
        else:
            return len(self.eval_idx_list)
