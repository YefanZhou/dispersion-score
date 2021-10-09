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
from auxiliary.my_utils import plant_seeds, load_config, chunks
from auxiliary.argument_parser import parser
from eval.metric import ChamferDistanceL2, compute_diff_ptcloud_dismatrix_batch
from eval.eval_utils import get_logger
import numpy as np

opt = parser()
opt.device = torch.device("cuda")
res_path = opt.dir_name
Path(res_path).mkdir(parents=True, exist_ok=True)
#proc_logger = get_logger("process", res_path, "process.log")

nviews_dic = {"train":opt.nviews_train, "test":opt.nviews_test}
plant_seeds(opt.manual_seed)

train_dataset = ShapeNet(opt, train=True, num_image_per_object=nviews_dic['train']) 

test_dataset = ShapeNet(opt, train=False, num_image_per_object=nviews_dic['test']) 

train_loader = torch.utils.data.DataLoader(train_dataset, 
                            batch_size=1, shuffle=False, num_workers=opt.workers)
test_loader = torch.utils.data.DataLoader(test_dataset, 
                            batch_size=1, shuffle=False, num_workers=opt.workers)

chamfer = ChamferDistanceL2().to(opt.device)

# results_dic = {'train':{'index':[], 'path':[]}, 'test':{'index':[], 'path':[]}, 'loss':[],
#                 'train_batch_size':opt.batch_size}

train_pts_set = torch.zeros(len(train_dataset), opt.number_points, 3)
test_pts_set = torch.zeros(len(test_dataset), opt.number_points, 3)
train_pts_pathset = []
test_pts_pathset =  []
train_cam_rotmatset = torch.zeros(len(train_dataset), 3, 3)
test_cam_rotmatset = torch.zeros(len(test_dataset), 3, 3)

for index, train_sample in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
    train_points, train_cam_rotmat, train_pts_path = train_sample['points'], train_sample['cam_rotmat'], \
                                                        train_sample['pointcloud_path']
    train_pts_set[[index]] = train_points
    train_cam_rotmatset[[index]] = train_cam_rotmat
    train_pts_pathset.append(train_pts_path)

for index, test_sample in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
    test_points, test_cam_rotmat, test_pts_path = test_sample['points'], test_sample['cam_rotmat'], \
                                                        test_sample['pointcloud_path']
    test_pts_set[[index]] = test_points
    test_cam_rotmatset[[index]] = test_cam_rotmat
    test_pts_pathset.append(test_pts_path)
    
train_pts_set = train_pts_set.to(opt.device)
test_pts_set = test_pts_set.to(opt.device)
train_cam_rotmatset = train_cam_rotmatset.to(opt.device)
test_cam_rotmatset = test_cam_rotmatset.to(opt.device)

if opt.mode == 'viewer':
    train_pts_set = torch.bmm(train_pts_set, torch.transpose(train_cam_rotmatset, 1, 2)) 
    test_pts_set = torch.bmm(test_pts_set, torch.transpose(test_cam_rotmatset, 1, 2)) 

opt.logger.info(f"train pts size: {len(train_pts_pathset)}")
opt.logger.info(f"test pts size: {len(test_pts_pathset)}")


dis_mat = compute_diff_ptcloud_dismatrix_batch(test_pts_set, train_pts_set, chamfer, 
        batch_size=opt.batch_size, device=opt.device, logger=opt.logger)

dis_mat = dis_mat.cpu().numpy()

np.save(os.path.join(res_path, 
    f"{opt.mode}_{opt.network}_trainnv{opt.nviews_train}.npy"),
    dis_mat)



# for test_index, test_sample in enumerate(test_loader):
#     min_loss = float('inf')
#     min_index = -1
#     min_train_path = ''
#     test_points, test_cam_rotmat, test_pts_path = test_sample['points'], \
#                             test_sample['cam_rotmat'], test_sample['pointcloud_path']
#     results_dic['test']['index'].append(test_index)
#     results_dic['test']['path'].append(test_pts_path)

#     if opt.mode == 'viewer':
#         test_points = torch.bmm(test_points, torch.transpose(test_cam_rotmat, 1, 2))
#         #test_points = test_points.expand(opt.batch_size, test_points.shape[1], test_points.shape[2])
#         test_points = test_points.repeat(opt.batch_size, 1, 1)
#     for train_index, train_sample in enumerate(train_loader):
#         train_points, train_cam_rotmat, train_pts_path = train_sample['points'], \
#                             train_sample['cam_rotmat'], train_sample['pointcloud_path']
#         if opt.mode == 'viewer':
#             train_points = torch.bmm(train_points, torch.transpose(train_cam_rotmat, 1, 2))
#         test_points = test_points.to(opt.device)
#         train_points = train_points.to(opt.device)

    
#         losses = chamfer(test_points, train_points, 'list')
#         new_loss, batch_index = torch.min(losses, dim=0)
#         new_loss = new_loss.detach().cpu().item()
#         batch_index = batch_index.detach().cpu().item()
#         if new_loss < min_loss:
#             min_loss = new_loss
#             min_index = (train_index, batch_index)
#             min_train_path = train_pts_path[batch_index]
#         # new_loss = chamfer(test_points, train_points).detach().cpu().item()
#         # if new_loss < min_loss:
#         #     min_index = train_index
#         #     min_loss = new_loss
#         #     min_train_path = train_pts_path
        
#         pbar.update(1)
#     results_dic['train']['index'].append(min_index)
#     results_dic['train']['path'].append(min_train_path)
#     results_dic['loss'].append(min_loss)





#        print(train_points.shape)
#        print(test_points.shape)

        #print(chamfer(test_points, train_points, 'list')[0].detach().cpu().item())   #.detach().cpu().item()


