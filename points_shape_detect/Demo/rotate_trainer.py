#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../udf-generate")

from points_shape_detect.Module.rotate_trainer import RotateTrainer


def demo():
    model_file_path = "./output/20221125_14:15:05/model_best.pth"
    model_file_path = ""
    resume_model_only = True
    print_progress = True

    rotate_trainer = RotateTrainer()
    rotate_trainer.loadModel(model_file_path, resume_model_only)
    #  rotate_trainer.testTrain()
    rotate_trainer.train(print_progress)
    return True
