#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import open3d as o3d
from random import randint

from pointr_detect.Data.bbox import BBox

from pointr_detect.Method.bbox import getOpen3DBBox, getOpen3DBBoxFromBBox


def getPCDFromPointArray(point_array, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_array)

    if color is not None:
        colors = np.array([color for _ in range(point_array.shape[0])],
                          dtype=float) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def renderPointArrayWithUnitBBox(point_array):
    unit_bbox = getOpen3DBBox()

    if isinstance(point_array, np.ndarray):
        pcd = getPCDFromPointArray(point_array)
    else:
        pcd = getPCDFromPointArray(point_array.detach().cpu().numpy())

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


def renderPredictBBox(data):
    if 'bbox' not in data['predictions'].keys():
        print("[ERROR][render::renderPredictBBox]")
        print("\t please save bbox during model running!")
        return False
    if 'center' not in data['predictions'].keys():
        print("[ERROR][render::renderPredictBBox]")
        print("\t please save center during model running!")
        return False
    if 'dense_points' not in data['predictions'].keys():
        print("[ERROR][render::renderPredictBBox]")
        print("\t please save dense_points during model running!")
        return False

    pcd_list = []

    if 'bbox' in data['inputs'].keys():
        gt_bbox_list = data['inputs']['bbox'][0].cpu().numpy().reshape(2, 3)
        gt_bbox = BBox.fromList(gt_bbox_list)
        open3d_gt_bbox = getOpen3DBBoxFromBBox(gt_bbox, [0, 255, 0])
        pcd_list.append(open3d_gt_bbox)

    if 'center' in data['inputs'].keys():
        gt_center = data['inputs']['center'][0].cpu().numpy().reshape(1, 3)
        gt_center_pcd = getPCDFromPointArray(gt_center, [0, 255, 0])
        pcd_list.append(gt_center_pcd)

    dense_points = data['predictions']['dense_points'][0].detach().cpu().numpy(
    )
    dense_points_pcd = getPCDFromPointArray(dense_points, [0, 0, 255])
    pcd_list.append(dense_points_pcd)

    bbox_list = data['predictions']['bbox'][0].detach().cpu().numpy().reshape(
        2, 3)
    bbox = BBox.fromList(bbox_list)
    open3d_bbox = getOpen3DBBoxFromBBox(bbox, [255, 0, 0])
    pcd_list.append(open3d_bbox)

    center = data['predictions']['center'][0].detach().cpu().numpy().reshape(
        1, 3)
    center_pcd = getPCDFromPointArray(center, [255, 0, 0])
    pcd_list.append(center_pcd)

    o3d.visualization.draw_geometries(pcd_list)
    return True
