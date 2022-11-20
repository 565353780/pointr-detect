#!/usr/bin/env python
# -*- coding: utf-8 -*-

from points_shape_detect.Module.trainer import Trainer


def demo():
    model_file_path = "./output/20221120_17:29:10/model_best.pth"
    model_file_path = ""
    print_progress = True

    trainer = Trainer()
    trainer.loadModel(model_file_path)
    #  trainer.testTrain()
    trainer.train(print_progress)
    return True
