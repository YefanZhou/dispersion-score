import os
import sys
import torch
import random
import tqdm
import time
import logging
from easydict import EasyDict
from pathlib import Path
from os.path import join
sys.path.append(join(os.path.dirname(os.path.abspath(__file__)), "../"))
from dataset.dataset_shapenet_views import ShapeNet
from dataset.vc_generator import plot
from auxiliary.my_utils import plant_seeds, load_config, chunks, visuaize_pts, image_grid
from auxiliary.argument_parser import parser
from eval.metric import ChamferDistanceL2, compute_ptcloud_dismatrix_batch
from eval.eval_utils import get_logger
import numpy as np

opt = parser()
opt.device = torch.device("cuda")
res_path = opt.dir_name
Path(res_path).mkdir(parents=True, exist_ok=True)
nviews_dic = {"train":opt.nviews_train, "test":opt.nviews_test}
plant_seeds(opt.manual_seed)

train_dataset = ShapeNet(opt, train=True, num_image_per_object=nviews_dic['train']) 
test_dataset = ShapeNet(opt, train=False, num_image_per_object=nviews_dic['test']) 

train_loader = torch.utils.data.DataLoader(train_dataset, 
                            batch_size=1, shuffle=False, num_workers=opt.workers)
test_loader = torch.utils.data.DataLoader(test_dataset, 
                            batch_size=1, shuffle=False, num_workers=opt.workers)

chamfer = ChamferDistanceL2().to(opt.device)

results_dic = {'train':{'index':[], 'path':[]}, 'test':{'index':[], 'path':[]}, 'loss':[],
                'train_batch_size':opt.batch_size}

train_pts_set = torch.zeros(len(train_dataset), opt.number_points, 3)
train_pts_pathset = []
train_cam_rotmatset = torch.zeros(len(train_dataset), 3, 3)

for index, train_sample in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
    train_points, train_cam_rotmat, train_pts_path = train_sample['points'], train_sample['cam_rotmat'], \
                                                        train_sample['pointcloud_path']
    train_pts_set[[index]] = train_points
    train_pts_pathset.append(train_pts_path)
    train_cam_rotmatset[[index]] = train_cam_rotmat

train_pts_set = train_pts_set.to(opt.device)
train_cam_rotmatset = train_cam_rotmatset.to(opt.device)

if opt.mode == 'viewer':
    train_pts_set = torch.bmm(train_pts_set, torch.transpose(train_cam_rotmatset, 1, 2)) 

dismat = compute_ptcloud_dismatrix_batch(train_pts_set, train_pts_set, chamfer, batch_size=opt.batch_size, 
device=opt.device, logger=opt.logger)

dismat = dismat.cpu().numpy()
np.save(os.path.join(res_path, f"{opt.mode}_trainnv{opt.nviews_train}_cls13{opt.shapenet13}.npy"), dismat)


# sampling and testing viewer-center point cloud 

# sample_img_list = []
# sample_pts_list = []
# train_indexes = random.sample(range(train_pts_set.shape[0]), 5)
# train_indexes.sort()
#sample_train_pts = train_pts_set[train_indexes]
# if index in train_indexes:
#     sample_img_list.append(train_sample['image'][0])
# for i in range(len(train_indexes)):
#     plot(sample_train_pts[i], sample_img_list[i].permute(1, 2, 0), 
#     title=f'test_{opt.mode}_{i}', ifsave=True, path=f'log/visualization/test_{opt.mode}_{i}_shapenet{opt.shapenet13}.png',
#     scale=0.8)