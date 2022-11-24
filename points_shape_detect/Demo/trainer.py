#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../udf-generate")

from points_shape_detect.Module.trainer import Trainer


def demo():
    model_file_path = "./output/20221122_23:38:42/model_best.pth"
    model_file_path = "./output/20221124_13:47:06/model_best.pth"
    model_file_path = ""
    resume_model_only = True
    print_progress = True

    trainer = Trainer()
    trainer.loadModel(model_file_path, resume_model_only)
    trainer.testTrain()
    #  trainer.train(print_progress)
    return True
