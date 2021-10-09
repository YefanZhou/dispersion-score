

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
from PIL import Image
import time
import logging
from easydict import EasyDict
from pathlib import Path
from os.path import join
import torch.nn as nn
sys.path.append(join(os.path.dirname(os.path.abspath(__file__)), "../../"))
from eval.eval_utils import get_logger, dic_to_box, mean_std
from dataset.dataset_shapenet_views import ShapeNet
from dataset.image_aug.AutoAugment.autoaugment import ImageNetPolicy, CIFAR10Policy
from dataset.image_aug.AutoAugment.autoaugment_param import PolicySeq
from auxiliary.my_utils import plant_seeds, image_grid
from auxiliary.metric_parser import parser
from eval.metric import pairwise_distances_torch, cluster_eval, PerceptualEncoder
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--perceptual", action="store_true", default=False)
parser.add_argument("--vis", action="store_true", default=False)
parser.add_argument("--aug-type", default='augment_mag', type=str, choices=['grayscale', 'rgb', 'binary', 'aug', 'normalize', 'augment_mag', 'auto_aug'])
parser.add_argument("--policy-type", default='ImageNet', choices=['ImageNet',  'CIFAR10'])
parser.add_argument("--c_method", default='KMedoids', type=str, choices=['AP', 'KMedoids'])
parser.add_argument("--e_method", default='Inertia', type=str, choices=['Inertia'])
parser.add_argument("--mag_idx", default=0, type=int, help="magnitude level index")
parser.add_argument("--n_op", default=0, type=int, help="number of operations")
parser.add_argument("--prob", default=1, type=float, help="number of operations")
parser.add_argument("--n_cluster", default=50, type=int)
parser.add_argument("--pc", default=10, type=int)
parser.add_argument("--color_type", default='brightness', type=str, choices=['brightness', 'contrast', 'saturation', 'hue', 'random'])
parser.add_argument("--color_aug_factor", nargs='+', default=[0.5, 0.5, 0.5, 0], type=float)
parser.add_argument("--nepoch", type=int, default=1)
parser.add_argument("--sample_num", type=int, default=30)
parser.add_argument("--normalize", default=False, action="store_true")

args = parser.parse_args()
args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

img_folder = '../../dataset/data/sample_rendering/'
img_paths = glob.glob(join(img_folder,  '*', '*', '*', '00.png'))  #
#img_paths.sort()
print(len(img_paths))
sample_num = args.sample_num
start_idx = 0
idx_list = []
for _ in range(args.nepoch):
    for i in range(start_idx, start_idx + sample_num):
        idx_list.append(i)
print(idx_list)

seed_score_list = []
if args.aug_type == 'rgb':
    transforms = transforms.Compose([
        transforms.CenterCrop(127),
        transforms.Resize(size=224, interpolation=2),
        transforms.ToTensor(),
    ])
elif args.aug_type == 'augment_mag':
    transforms = transforms.Compose([
            transforms.CenterCrop(127),
            transforms.Resize(size=224, interpolation=2),
            lambda x: x.convert('RGB'),
            PolicySeq(mag_idx=args.mag_idx, prob=args.prob, n_op=args.n_op),
            transforms.ToTensor(),
    ])
elif args.aug_type == 'auto_aug':
    print("verbose")
    if args.policy_type == 'ImageNet':
        Policy = ImageNetPolicy
    elif args.policy_type == 'CIFAR10':
        Policy = CIFAR10Policy
    else:
        raise NotImplementedError()

    transforms = transforms.Compose([
            transforms.CenterCrop(127),
            transforms.Resize(size=224, interpolation=2),
            lambda x: x.convert('RGB'),
            Policy(),
            transforms.ToTensor(),
    ])


percep_encoder = PerceptualEncoder().to(args.device)
for seed in [1, 2, 3, 4, 5]:       #  

    plant_seeds(manual_seed=seed)

    dic_box_list = [] 
    images = None
    for i in idx_list:
        tmp_image = Image.open(img_paths[i])
        tmp_image = transforms(tmp_image)
        tmp_image = tmp_image[:3, :, :]
        #plt.imshow(tmp_image.permute(1, 2, 0))
        #plt.show()
        tmp_image = tmp_image.unsqueeze(0)
        if images is None:
            images = tmp_image
        else:
            images = torch.cat((images, tmp_image), dim=0)

    #assert images.shape == torch.Size([sample_num, 3, 224, 224])
    #print("images", torch.max(images), torch.min(images))
    images = images.to(args.device)
    if args.normalize:
        batch_size, c, height, width = images.shape
        images = images.view(batch_size, -1)
        images -= images.min(1, keepdim=True)[0]
        images /= images.max(1, keepdim=True)[0]
        images = images.view(batch_size, c, height, width)

    if args.perceptual:
        _, _, h_relu_3_3, _ = percep_encoder(images)
        #relu_3_3, _ = torch.max(h_relu_3_3, dim=1, keepdim=True)
        relu_3_3 = h_relu_3_3
        feature = relu_3_3
        #---------------------
        # batch_size, _, height, width = feature.shape
        # feature = feature.view(batch_size, -1)
        # feature -= feature.min(1, keepdim=True)[0]
        # feature /= feature.max(1, keepdim=True)[0]
        # feature = feature.view(batch_size, 1, height, width)
    else:
        feature = images


    #print("feature", torch.max(feature[0]), torch.min(feature[0]))
    print(feature.shape)
    feature_flatten = feature.view(feature.shape[0], -1)
    dm = pairwise_distances_torch(feature_flatten)
    dm = dm.numpy()
    score, matrix_part = cluster_eval(c_method=args.c_method, e_method=args.e_method, n_cluster=args.n_cluster, pc=args.pc, distance_matrix=dm, seed=0,  mean=False)
    seed_score_list.append(score)
    #print("Distance Matrix Mean", np.mean(dm))
    if args.vis:
        demo = feature.permute(0, 2, 3, 1)
        image_grid(demo, rows=5, cols=6, title=f'{args.aug_type}_demo_mag{args.mag_idx}_nop{args.n_op}_prob{args.prob}_perp{args.perceptual}_seed{seed}\n{score:.4f}')
        plt.tight_layout()
        plt.show()
        #plt.savefig(f'results/{args.aug_type}/{args.aug_type}_demo_mag{args.mag_idx}_nop{args.n_op}_prob{args.prob}_perp{args.perceptual}_seed{seed}.png')

    # print(feature.shape)
    
    #print(seed, args.aug_type, f"{score:.5f}")
    
score_mean_std = mean_std(seed_score_list)
print(f"Score: {score_mean_std[0]:.5f} {score_mean_std[1]:.5f}")








