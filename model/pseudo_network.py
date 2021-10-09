
import os
import sys
import torch
import numpy as np
import torch.nn as nn
from os.path import join
import matplotlib.pyplot as plt
sys.path.append('../')
import dataset.pointcloud_processor as pointcloud_processor
from auxiliary.my_utils import visuaize_pts

class Generator(nn.Module):
    def __init__(self, data, batch_size):
        self.data = data.split(batch_size)
        self.index = 0
    def __call__(self, image, train=False):
        self.index = self.index + 1

        return self.data[self.index - 1]

class CondGenerator(nn.Module):
    def __init__(self, base_path, points_num=2500, normalization="UnitBall"):
        self.base_path = base_path 
        self.points_num = points_num
        if normalization == "UnitBall":
            print(normalization)
            self.normalization_function = pointcloud_processor.Normalization.normalize_unitL2ball_functional
        elif normalization == "BoundingBox":
            self.normalization_function = pointcloud_processor.Normalization.normalize_bounding_box_functional
        else:
            self.normalization_function = pointcloud_processor.Normalization.identity_functional
    def __call__(self, cat_list, id_list):
        assert len(cat_list) == len(id_list)
        init_pts = np.zeros((len(cat_list), self.points_num, 3))
        idx = 0
        for cat, ids in zip(cat_list, id_list):
            init_pts[idx] = np.load(join(self.base_path, cat, f'{ids}.npy'))
            idx += 1

        ptcloud = torch.from_numpy(init_pts).float()
        ptcloud = self.normalization_function(ptcloud)

        #ptcloud = pointcloud_processor.Normalization(ptcloud).normalize_unitL2ball()

        return ptcloud



if __name__ == '__main__':
    '04256520 '
    'e61751c485c937fa717300ec93e59c63'
    'ffc0bf43dccf71865257cc5c33f9e8a3'
    
    ptcloud = pix2mesh(['04256520', '04256520','02828884','02828884'], ['e61751c485c937fa717300ec93e59c63','ffc0bf43dccf71865257cc5c33f9e8a3', 'ff7a08ac16701dcec0277fc71766d822', 'f560aae80e66f50585ef4356ea4ff622'])

    visuaize_pts(ptcloud, 'test_pix2mesh', elev=0, azim=0)
    plt.savefig('../dataset/data/pix2mesh_peusdomodel_test/test_1.png')

    print(ptcloud.shape)
