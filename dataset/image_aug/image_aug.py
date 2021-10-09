

import glob
from matplotlib import pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import torch.nn as nn
import os
import sys
import torch
import random
from torchvision import transforms
import tqdm
from PIL import Image, ImageFilter
import time
import logging
from easydict import EasyDict
from pathlib import Path
from os.path import join
import torch.nn as nn
sys.path.append(join(os.path.dirname(os.path.abspath(__file__)), "../../"))
from eval.eval_utils import get_logger, dic_to_box, mean_std
from dataset.dataset_shapenet_views import ShapeNet
from dataset.image_aug.AutoAugment.autoaugment import ImageNetPolicy, CIFAR10Policy, CombinedPolicy
from auxiliary.my_utils import plant_seeds, image_grid
from auxiliary.metric_parser import parser
from eval.metric import pairwise_distances_torch, cluster_eval
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--aug-type", default='rgb', type=str, choices=['grayscale', 'rgb', 'binary', 'aug', 'normalize', 'autoaugment'])
parser.add_argument("--c_method", default='KMedoids', type=str, choices=['AP', 'KMedoids'])
parser.add_argument("--e_method", default='Inertia', type=str, choices=['Inertia'])
parser.add_argument("--n_cluster", default=50, type=int)
parser.add_argument("--pc", default=10, type=int)
parser.add_argument("--color_type", default='brightness', type=str, choices=['brightness', 'contrast', 'saturation', 'hue', 'random'])
parser.add_argument("--color_aug_factor", nargs='+', default=[0.5, 0.5, 0.5, 0], type=float)

args = parser.parse_args()

img_folder = '../../dataset/data/ShapeNetV1Renderings/'
img_paths = glob.glob(join(img_folder,  '*', '*', '*', '00.png'))  #
print(len(img_paths))
sample_num = 30
start_idx = 0

seed_score_list = []
for seed in [1, 2, 3, 4, 5]:    #
    plant_seeds(manual_seed=seed)

    rgb_transforms = transforms.Compose([
        transforms.CenterCrop(127),
        transforms.Resize(size=224, interpolation=2),
        transforms.ToTensor(),
    ])

    grayscale_transforms = transforms.Compose([
        transforms.CenterCrop(127),
        transforms.Resize(size=224, interpolation=2),
        transforms.Grayscale(num_output_channels=3), 
        transforms.ToTensor(),
    ])

    b, c, s, h = args.color_aug_factor

    coloraug_transforms = transforms.Compose([
        transforms.CenterCrop(127),
        transforms.Resize(size=224, interpolation=2),
        transforms.ColorJitter(brightness=b, 
                contrast=c, saturation=s, hue=h),
        transforms.ToTensor(),
    ])

    normalize_transforms = transforms.Compose([
            transforms.CenterCrop(127),
            transforms.Resize(size=224, interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
    ])

    autoaug_transforms = transforms.Compose([
            lambda x: x.convert('RGB'),
            transforms.CenterCrop(127),
            transforms.Resize(size=224, interpolation=2),
            ImageNetPolicy(),
            transforms.ToTensor(),
    ])
    # factor_list = []
    # for factor in np.arange(-0.5, 0.5, 0.2):
    #     factor_list.append(factor)
    #for h in factor_list:

    dic_box_list = [] #
    
    for i in range(start_idx, start_idx + sample_num):
        tmp_image = Image.open(img_paths[i])
        #tmp_image = cv2.imread(img_paths[i])
        if args.aug_type == 'grayscale':
            tmp_image = grayscale_transforms(tmp_image)
        elif args.aug_type == 'rgb':
            tmp_image = rgb_transforms(tmp_image)
        elif args.aug_type == 'binary':
            tmp_image = transforms.CenterCrop(127)(tmp_image)
            tmp_image = transforms.Resize(size=224, interpolation=2)(tmp_image)
            tmp_image = np.array(tmp_image)
            tmp_image = tmp_image[:, :, 3] > 0
            tmp_image = tmp_image.astype(float)
            tmp_image = transforms.ToTensor()(tmp_image)
            tmp_image = tmp_image.repeat(3, 1, 1)
        elif args.aug_type == 'aug':
            tmp_image = coloraug_transforms(tmp_image)
        elif args.aug_type == 'normalize':
            tmp_image = transforms.CenterCrop(127)(tmp_image)
            tmp_image = transforms.Resize(size=224, interpolation=2)(tmp_image)
            tmp_image = transforms.ToTensor()(tmp_image)
            tmp_image = tmp_image[:3, :, :]
        elif args.aug_type == 'autoaugment':
            tmp_image = autoaug_transforms(tmp_image)


        tmp_image = tmp_image[:3, :, :]
        if i == start_idx:
            images = tmp_image.unsqueeze(0)
        else:
            images = torch.cat((images, tmp_image.unsqueeze(0)), dim=0)

    assert images.shape == torch.Size([sample_num, 3, 224, 224])

    # demo_images = images.permute(0, 2, 3, 1)
    # image_grid(demo_images, rows=5, cols=6, title=f'{args.aug_type}')
    # plt.tight_layout()
    # #plt.show()
    # plt.savefig(f'results/{args.aug_type}_{args.color_type}_demo_seed{seed}.png')
    
    images = images.view(images.shape[0], -1)
    dm = pairwise_distances_torch(images)
    dm = dm.numpy()

    score, matrix_part = cluster_eval(c_method=args.c_method, e_method=args.e_method, n_cluster=args.n_cluster, pc=args.pc, distance_matrix=dm, seed=0,  mean=False)
    seed_score_list.append(score)
    #print(seed, args.aug_type, f"{score:.5f}")
    

        # dm_stat_dic = {"lower_quat":np.percentile(dm, 25), 
        #                 "up_quat": np.percentile(dm, 75), 
        #                 "mean": np.mean(dm),
        #                 "min": np.min(dm),
        #                 "max": np.max(dm),
        #                 "median": np.median(dm),
        #                 "std": np.std(dm)}
        # #print(dm_stat_dic)
        # dic_box_list.append(dic_to_box(dm_stat_dic)[0])


score_mean_std = mean_std(seed_score_list)
print(f"{score_mean_std[0]:.5f} {score_mean_std[1]:.5f}")


# ax = plt.subplot(111)
# ax.bxp(dic_box_list, showfliers=False)
# ax.set_ylim([0, 0.45])
# ax.set_title("DM distribution")
# plt.show()

# plt.plot(factor_list, adjust_score_list)
# plt.title(f"RGB {args.color_type}")
# plt.xlabel("Factor")
# plt.ylabel("Inertia")
# plt.show()










