#!/usr/bin/env python
# -*- coding: utf-8 -*-

from points_shape_detect.Test.iou import test as test_iou
from points_shape_detect.Test.trans import test as test_trans

if __name__ == "__main__":
    test_iou()
    test_trans()
