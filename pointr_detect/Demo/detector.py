#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pointr_detect.Module.detector import Detector


def demo():
    model_file_path = "/home/chli/chLi/PoinTr/pointr_training_from_scratch_c55_best.pth"
    detector = Detector(model_file_path)
    detector.detectPointArray(None)
    return True
