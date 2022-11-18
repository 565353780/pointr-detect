#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import open3d as o3d

from pointr_detect.Method.sample import seprate_point_cloud
from pointr_detect.Method.move import moveToOrigin
from pointr_detect.Method.render import renderPointArrayWithUnitBBox

from pointr_detect.Module.detector import Detector


def demo():
    model_file_path = "/home/chli/chLi/PoinTr/pointr_training_from_scratch_c55_best.pth"

    detector = Detector(model_file_path)

    points = np.random.randn(300, 3) - 0.5
    points = moveToOrigin(points)
    data = detector.detectPointArray(points)
    print(data['predictions'].keys())
    renderPointArrayWithUnitBBox(data['predictions']['dense_points'][0])
    return True


def demo_mesh():
    model_file_path = "/home/chli/chLi/PoinTr/pointr_training_from_scratch_c55_best.pth"
    shapenet_model_file_path = "/home/chli/chLi/ShapeNet/Core/ShapeNetCore.v2/" + \
        "02691156/1a04e3eab45ca15dd86060f189eb133" + \
        "/models/model_normalized.obj"

    detector = Detector(model_file_path)

    assert os.path.exists(shapenet_model_file_path)
    mesh = o3d.io.read_triangle_mesh(shapenet_model_file_path)
    pcd = mesh.sample_points_uniformly(8192)
    points = np.array(pcd.points).reshape(1, -1, 3)
    partial, _ = seprate_point_cloud(points, 8192, 4096)
    partial = moveToOrigin(partial)
    data = detector.detectPointArray(partial)
    print(data['predictions'].keys())
    renderPointArrayWithUnitBBox(data['predictions']['dense_points'][0])
    return True
