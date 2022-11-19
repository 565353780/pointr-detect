#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import torch.utils.data as data

from pointr_detect.Data.io import IO

from pointr_detect.Method.trans import normalizePointArray

class ShapeNet55Dataset(data.Dataset):

    def __init__(self):
        self.data_root = '../PoinTr/data/ShapeNet55-34/ShapeNet-55'
        self.pc_path = '/home/chli/chLi/PoinTr/ShapeNet55/shapenet_pc'
        self.subset = 'test'
        self.data_list_file = os.path.join(self.data_root,
                                           f'{self.subset}.txt')

        print(f'[DATASET] Open file {self.data_list_file}')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()

        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print(f'[DATASET] {len(self.file_list)} instances were loaded')

    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = {'inputs': {}, 'predictions': {}, 'losses': {}, 'logs': {}}

        data['inputs']['taxonomy_id'] = sample['taxonomy_id']
        data['inputs']['model_id'] = sample['model_id']

        point_array = IO.get(os.path.join(
            self.pc_path, sample['file_path'])).astype(np.float32)
        point_array = normalizePointArray(point_array)
        data['inputs']['point_array'] = torch.from_numpy(point_array).float()
        return data

    def __len__(self):
        return len(self.file_list)
