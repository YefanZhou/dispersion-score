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
from auxiliary.my_utils import plant_seeds, load_config, chunks, visuaize_pts
from auxiliary.argument_parser import parser
from eval.metric import ChamferDistanceL2, compute_diff_ptcloud_dismatrix_batch
from eval.eval_utils import get_logger
import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--base-dir", type=str, default='log/benchmark/20210202_2055_viewer_clustering_seed1', 
                    help='')
parser.add_argument("--dismat-fname", type=str, default='viewer_trainnv1_cls13True.npy', 
                    help='')   
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8) 
parser.add_argument("--views_search", action="store_true", help="increment views to train")
parser.add_argument("--mode", type=str, default="viewer", choices=['viewer', 'object'])
parser.add_argument("--nviews_train", type=int, default=1, help='num of view per shape for training')
parser.add_argument("--nviews_test", type=int, default=1, help='num of view per shape for test')    

parser.add_argument('--normalization', type=str, default="UnitBall",
                    choices=['UnitBall', 'BoundingBox', 'Identity'])
parser.add_argument("--shapenet13", action="store_true", help="Load 13 usual shapenet categories")
parser.add_argument("--SVR", action="store_true", help="Single_view Reconstruction")
parser.add_argument("--sample", action="store_false", help="Sample the input pointclouds")
parser.add_argument('--class_choice', nargs='+', default=["table", "car"], type=str)
parser.add_argument('--number_points', type=int, default=2500, help='Number of point sampled on the object during training, and generated by atlasnet')
parser.add_argument('--number_points_eval', type=int, default=2500,
                    help='Number of points generated by atlasnet (rounded to the nearest squared number) ')
parser.add_argument("--manual_seed", type=int, default=1, help="if seed > 0, then fixed, if < 0 then random")
parser.add_argument("--img_aug", action="store_true", help="apply image augmentation like random crop")
parser.add_argument("--img_aug_type", type=str, default='rgb', choices=['grayscale', 'rgb', 'binary', 'color_aug', 'color_aug_random', 'autoaugment'])
parser.add_argument("--autoaug_type", type=str, default='ImageNet', choices=['ImageNet', 'CIFAR10', 'Seq', 'RGB', 'SVHN'])
parser.add_argument("--color_aug_factor", nargs='+', type=float, default=[1.0, 1.0, 1.0, 1.0], help='brightness, contrast, saturation, hue')
parser.add_argument("--mag_idx", default=0, type=int, help="magnitude level index")
parser.add_argument("--n_op", default=0, type=int, help="number of operations")
parser.add_argument("--prob", default=1, type=float, help="prob")
parser.add_argument("--demo", action="store_true", help="run demo autoencoder or single-view")
parser.add_argument('--no_compile_chamfer', action="store_true", help="compile c++ version chamfer distance")
parser.add_argument('--n_clusters', type=int, default=500, help="number of clusters")
parser.add_argument('--cluster_res_path', type=str, default=None, help="number of clusters")

parser.add_argument('--lrate', type=float, default=0.001, help='learning rate')
parser.add_argument('--nepoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--decay_step', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--batch_size_test', type=int, default=32, help='input batch size')
parser.add_argument('--description', type=str, default="", help='descript this training')
parser.add_argument('--optimize_option', type=int, default=0, help='')
parser.add_argument('--pretrained', action='store_false', default=True, help='')
parser.add_argument('--test', action='store_true', default=False, help='')
parser.add_argument('--network_dir', type=str, default='210205_0221_viewer_optimize1_pretrainedTrue_decay30', help='')
parser.add_argument('--network', type=str, default='clustering', help='')
parser.add_argument('--dismat_path', type=str, 
                    default="log/benchmark/20210202_2351_viewer_oraclenn_seed1/viewer_oraclenn_trainnv1.npy", 
                    help="path to distance matrix for Oracle NN or Clustering")

opt = parser.parse_args()
opt = EasyDict(opt.__dict__)

opt.device = torch.device("cuda")
res_path = opt.network_dir

Path(res_path).mkdir(parents=True, exist_ok=True)
opt.logger = get_logger("eval", res_path, "eval.log")

nviews_dic = {"train":opt.nviews_train, "test":opt.nviews_test}
plant_seeds(opt.manual_seed)

dismat = np.load(opt.dismat_path)
losses = np.min(dismat, axis=1)

opt.logger.info(f"Predict Shape: {dismat.shape[0]}, Chamfer Distance {np.mean(losses):6f}")

train_indexes = np.argmin(dismat, axis=1)

# for idx, index in enumerate(train_indexes):
#     print(idx, index)
#     print(dismat[idx][index], np.min(dismat[idx]))
#     print("--------------------")
    
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
    
train_cam_rotmatset = train_cam_rotmatset.to(opt.device)
test_cam_rotmatset = test_cam_rotmatset.to(opt.device)
train_pts_set = train_pts_set.to(opt.device)
test_pts_set = test_pts_set.to(opt.device)

if opt.mode == 'viewer':
    train_pts_set = torch.bmm(train_pts_set, torch.transpose(train_cam_rotmatset, 1, 2)) 
    test_pts_set = torch.bmm(test_pts_set, torch.transpose(test_cam_rotmatset, 1, 2)) 

test_sample_pts = test_pts_set[:train_indexes.shape[0]] 
train_sample_pts = train_pts_set[train_indexes]

test_sample_pts = test_sample_pts.cpu()
train_sample_pts = train_sample_pts.cpu()

assert train_sample_pts.shape[0] == test_sample_pts.shape[0]



np.save(join(res_path, "prediction.npy"), train_sample_pts)
path = join(res_path,  "prediction.npy")
opt.logger.info(f"SAVING PREDICITON TO {path}")


##visualization
# visuaize_pts(test_sample_pts, "test_gt", rows=2, cols=5, elev=0, azim=0)
# plt.savefig(os.path.join("log/visualization", "oracle_test_gt.png"))
# visuaize_pts(train_sample_pts, "oracle_pred", rows=2, cols=5, elev=0, azim=0)
# plt.savefig(os.path.join("log/visualization", "oracle_pred.png"))