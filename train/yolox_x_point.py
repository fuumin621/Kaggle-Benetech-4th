
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
        self.input_size = (960, 960)
        self.data_num_workers = 8
        
        self.data_dir = "./yolox_point"
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        self.output_dir = "./../output/YOLOX_outputs"
        
        ## クラス数の変更
        self.num_classes = 1
        
        self.max_epoch = 50
        self.no_aug_epochs = 15

        ## 評価間隔を変更（初期では10epochごとにしか評価が回らない）
        self.eval_interval = 2
