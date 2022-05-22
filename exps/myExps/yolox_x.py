#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import platform
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.platform_system = platform.system()
        self.data_dir = "W:/tanimoto.j/dataset/flir/FLIR_ADAS_1_3"
        self.train_ann = "train/thermal_annotations.json"
        self.val_ann = "val/thermal_annotations.json"
        if self.platform_system != "Windows":
            self.data_dir = "/work/tanimoto.j/dataset/flir/FLIR_ADAS_1_3/"
            self.train_ann = "train/thermal_annotations.json"
            self.val_ann = "val/thermal_annotations.json"
