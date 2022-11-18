#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import open3d as o3d

from pointr_detect.Method.bbox import getOpen3DBBox


def renderPointArrayWithUnitBBox(point_array):
    unit_bbox = getOpen3DBBox()

    pcd = o3d.geometry.PointCloud()
    if isinstance(point_array, np.ndarray):
        pcd.points = o3d.utility.Vector3dVector(point_array)
    elif isinstance(point_array, torch.Tensor):
        pcd.points = o3d.utility.Vector3dVector(
            point_array.detach().cpu().numpy())

    o3d.visualization.draw_geometries([unit_bbox, pcd])
    return True
