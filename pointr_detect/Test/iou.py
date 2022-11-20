#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np

from pointr_detect.Loss.ious import IoU, CIoU, DIoU, EIoU, GIoU


def demo():
    bbox1 = np.array([0, 0, 0, 1, 1, 1])
    bbox2 = np.array([0.2, 0.2, 0.2, 1.2, 1.2, 1.2])
    bbox1 = torch.from_numpy(bbox1).float()
    bbox2 = torch.from_numpy(bbox2).float()

    for i in range(10):
        bbox2 = bbox1.clone()
        bbox2[5] = i

        print("================================")
        iou = IoU(bbox1, bbox2)
        print("IoU", iou)
        iou = CIoU(bbox1, bbox2)
        print("CIoU", iou)
        iou = DIoU(bbox1, bbox2)
        print("DIoU", iou)
        iou = EIoU(bbox1, bbox2)
        print("EIoU", iou)
        iou = GIoU(bbox1, bbox2)
        print("GIoU", iou)
    return True
