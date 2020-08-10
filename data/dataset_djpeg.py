"""
Created by Myung-Joon Kwon
corundum240@gmail.com
Aug 4, 2020
"""
import project_config
from data import AbstractDataset

import os
import numpy as np
import random
from PIL import Image
from pathlib import Path
import torch


class Djpeg(AbstractDataset.AbstractDataset):
    def __init__(self, crop_size, grid_crop, blocks: list, DCT_channels: int, tamp_list: str):
        """
        :param crop_size: (H,W) or None
        :param blocks:
        :param tamp_list: EX: "Splicing/data/Ca??_list.txt"
        :param read_from_jpeg: F=from original extension, T=from jpeg compressed image
        """
        super().__init__(crop_size, grid_crop, blocks, DCT_channels)
        self._root_path = project_config.dataset_paths['djpeg']
        with open(project_config.project_root / tamp_list, "r") as f:
            self.tamp_list = [t.strip().split(',') for t in f.readlines()]

    def get_tamp(self, index):
        assert 0 <= index < len(self.tamp_list), f"Index {index} is not available!"
        tamp_path = self._root_path / self.tamp_list[index][0]
        return self._create_tensor(tamp_path, int(self.tamp_list[index][1]))

