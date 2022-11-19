#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import open3d as o3d
from random import randint

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


def renderRebuildPatchPoints(data):
    if 'coarse_point_cloud' not in data['predictions'].keys():
        print("[ERROR][render::renderRebuildPatchPoints]")
        print("\t please save coarse_point_cloud during model running!")
        return False
    if 'rebuild_patch_points' not in data['predictions'].keys():
        print("[ERROR][render::renderRebuildPatchPoints]")
        print("\t please save rebuild_patch_points during model running!")
        return False

    coarse_point_cloud = data['predictions']['coarse_point_cloud'][0].detach(
    ).cpu().numpy()
    rebuild_patch_points = data['predictions']['rebuild_patch_points'][
        0].detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    all_points = []
    all_colors = []
    center_color = [255, 0, 0]
    for i in range(coarse_point_cloud.shape[0]):
        translate = [int(i / 10), i % 10, 0]
        color = [randint(0, 128), randint(0, 255), randint(0, 255)]
        query = [coarse_point_cloud[i][j] + translate[j] for j in range(3)]
        points = rebuild_patch_points[i]
        all_points.append(query)
        all_colors.append(center_color)
        for point in points:
            all_points.append([point[j] + translate[j] for j in range(3)])
            all_colors.append(color)

    all_points = np.array(all_points, dtype=float)
    all_colors = np.array(all_colors, dtype=float) / 255.0
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    o3d.visualization.draw_geometries([pcd])
    return True
