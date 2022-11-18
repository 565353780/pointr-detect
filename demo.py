#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pointr_detect.Demo.detector import \
    demo as demo_detect, \
    demo_mesh as demo_detect_partial_mesh
from pointr_detect.Demo.trainer import demo as demo_train

if __name__ == "__main__":
    #  demo_detect()
    #  demo_detect_partial_mesh()
    demo_train()
