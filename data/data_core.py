"""
Created by Myung-Joon Kwon
corundum240@gmail.com
July 8, 2020
"""


import torch
from torch.utils.data import Dataset

from data.dataset_djpeg import Djpeg



class SplicingDataset(Dataset):
    def __init__(self, crop_size, grid_crop, blocks=('RGB',), mode="train", DCT_channels=3, read_from_jpeg=False, class_weight=None):
        self.dataset_list = []
        if mode == "train":
            self.dataset_list.append(Djpeg(crop_size, grid_crop, blocks, DCT_channels, "data/train.txt"))
        elif mode == "valid":
            self.dataset_list.append(Djpeg(crop_size, grid_crop, blocks, DCT_channels, "data/val.txt"))
        else:
            raise KeyError("Invalid mode" + mode)
        if class_weight is None:
            self.class_weights = torch.FloatTensor([1.0, 1.0])
        else:
            self.class_weights = torch.FloatTensor(class_weight)
        self.crop_size = crop_size
        self.grid_crip = grid_crop
        self.blocks = blocks
        self.mode = mode
        self.read_from_jpeg = read_from_jpeg


    def __len__(self):
        # return 400
        return sum([len(lst) for lst in self.dataset_list])

    def __getitem__(self, index):
        it = 0
        while True:
            if index >= len(self.dataset_list[it]):  # not '>' :D
                index -= len(self.dataset_list[it])
                it += 1
                continue
            return self.dataset_list[it].get_tamp(index)

    def get_class_weight(self):
        c0 = 0
        c1 = 0
        # w = 0
        # h = 0
        for i in range(self.__len__()):
            it = 0
            while True:
                if i >= len(self.dataset_list[it]):
                    i -= len(self.dataset_list[it])
                    it += 1
                    continue
                _, mask = self.dataset_list[it].get_tamp(i)
                temp_c0 = torch.sum(mask == 0.0).item()
                temp_c1 = mask.shape[0]*mask.shape[1] - temp_c0
                c0 += temp_c0
                c1 += temp_c1
                # w += mask.shape[1]
                # h += mask.shape[0]
                break
        class_weight = 1 / torch.log(torch.tensor([c0, c1], dtype=torch.float))
        # portion weight [c1, c0] : [0.228743, 0.771257] * 2
        # log-pixels weight : [0.0431, 0.0462]
        # H, W average : 678.91, 902.21
        return class_weight
    # usage (in train.py) :
    # temp_dataset = splicing_dataset(None, False, blocks=('RGB',), mode='train')
    # class_weight = temp_dataset.get_class_weight()

    def get_info(self):
        s = ""
        for ds in self.dataset_list:
            s += (str(ds)+'('+str(len(ds))+') ')
        s += '\n'
        s += f"crop_size={self.crop_size}, grid_crop={self.grid_crip}, blocks={self.blocks}, mode={self.mode}, read_from_jpeg={self.read_from_jpeg}"
        return s





