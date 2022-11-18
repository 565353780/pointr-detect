#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pointr_detect.Module.trainer import Trainer


def demo():
    model_file_path = "/home/chli/chLi/PoinTr/pointr_training_from_scratch_c55_best.pth"

    trainer = Trainer(model_file_path)
    trainer.testTrain()
    trainer.train()
    return True
