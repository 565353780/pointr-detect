#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pointr_detect.Model.pointr import PoinTr

class Detector(object):
    def __init__(self):
        self.model = PoinTr()
        return
