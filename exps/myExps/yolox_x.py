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
        
        # --------------  training config --------------------- #
        self.warmup_epochs = 5
        self.max_epoch = 115
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 10
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]