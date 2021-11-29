"""Combine pred_sscore_v2.py and gt_cluster_score.py
"""
import os
import sys
import numpy as np
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
from auxiliary.my_utils import plant_seeds, load_config
from auxiliary.metric_parser import parser
from auxiliary.constant import DIM_TEMPLATE_DICT
from model.baseline_models import FoldNet, PSGN
from model.pseudo_network import Generator
from eval.metric import ChamferDistanceL2, compute_ptcloud_dismatrix_batch, pairwise_distances, cluster_eval
from eval.eval_utils import get_logger, mean_std
from model.model import EncoderDecoder
import auxiliary.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D

opt = parser()
###Mkdir and logger
opt.device = torch.device("cuda")

res_path = join(opt.dir_name, opt.res_folder)
Path(res_path).mkdir(parents=True, exist_ok=True)
proc_logger = get_logger("process", res_path, "process.log")
res_logger = get_logger("results", res_path, "sscore.log")
opt.logger = proc_logger
exp_opts = load_config(join(opt.trained_exp_dir, "opts.yaml"))
exp_opts = EasyDict(exp_opts)
if "img_aug_type" not in exp_opts:
    exp_opts.img_aug_type = opt.img_aug_type
    exp_opts.autoaug_type = opt.autoaug_type
    exp_opts.color_aug_factor = opt.color_aug_factor
    exp_opts.mag_idx = opt.mag_idx
    exp_opts.n_op = opt.n_op
    exp_opts.prob = opt.prob

if "encoder" not in exp_opts:
    exp_opts['encoder'] = 'resnet18'

exp_opts['test_augment'] = opt.test_augment

exp_opts.logger = proc_logger
exp_opts.device = torch.device("cuda")
if exp_opts.network != 'clustering' and exp_opts.network != 'oraclenn':
    exp_opts.dim_template = DIM_TEMPLATE_DICT[exp_opts.template_type]

nviews_dic = {"train":opt.nviews_train, "test":opt.nviews_test}
num_seed = max(len(opt.seed_list), 1)
sscore_collect = {}
eval_label_list = set()

for seed_idx in range(num_seed):
    if opt.seed_list:
        opt.seed = opt.seed_list[seed_idx]
    sscore_collect.update({str(opt.seed):{}})

    plant_seeds(opt.seed)
    ## train
    if opt.split == "train":
        dataset = ShapeNet(exp_opts, train=True, num_image_per_object=nviews_dic['train']) 
    elif opt.split == "test" or opt.split == "pred":
        dataset = ShapeNet(exp_opts, train=False, num_image_per_object=nviews_dic['test']) 
    else:
        raise NotImplementedError()


    if opt.rsample == 1:
        sample_num = len(dataset)
        opt.nsample = len(dataset)
    else:
        if opt.rsample != -1:
            opt.nsample = int(opt.rsample * len(dataset))
        subset_index = random.sample(range(len(dataset)), opt.nsample)
        dataset = torch.utils.data.Subset(dataset, subset_index)
        sample_num = len(subset_index)

    loader = torch.utils.data.DataLoader(dataset, batch_size=opt.pred_batch_size, shuffle=False, num_workers=8)
    data_list = []
    cat_list = []

    ##Loading Network
    if opt.split == 'pred':
        eval_loss = ChamferDistanceL2().to(opt.device)
        distChamfer = dist_chamfer_3D.chamfer_3DDist()
        if exp_opts.network == 'psgn':
            network = PSGN(num_points=exp_opts.number_points, 
                                bottleneck_size=exp_opts.bottleneck_size, 
                                hidden_neurons=exp_opts.hidden_neurons, 
                                pretrained_encoder=False,
                                remove_all_batchNorms=exp_opts.remove_all_batchNorms)
        elif exp_opts.network == 'atlasnet':
            network = EncoderDecoder(exp_opts)
        elif exp_opts.network == 'foldnet':
            network = FoldNet(template_type=exp_opts.template_type, 
                                    num_points=exp_opts.number_points,
                                    bottleneck_size=exp_opts.bottleneck_size, 
                                    hidden_neurons=exp_opts.hidden_neurons,
                                    pretrained_encoder=False,
                                    remove_all_batchNorms=exp_opts.remove_all_batchNorms)
        elif exp_opts.network == 'clustering' or exp_opts.network == 'oraclenn':
            data = np.load(os.path.join(opt.trained_exp_dir, 'prediction.npy'))
            data = torch.from_numpy(data).to(opt.device)
            network = Generator(data, opt.pred_batch_size)
        else:
            raise NotImplementedError(f"{exp_opts.network} is not implemented/imported")
        
        if exp_opts.network != 'clustering' and exp_opts.network != 'oraclenn':
            network.to(exp_opts.device)
            network.load_state_dict(torch.load(join(opt.trained_exp_dir, "network.pth"), map_location='cuda'))
            proc_logger.info(f"Network weights loaded from  {opt.trained_exp_dir}!")
            network.eval()

    pred_loss = 0.0

    with torch.set_grad_enabled(False): 
        for batch in tqdm.tqdm(loader, desc=f"loading {opt.split} {opt.type} data"):
            cat_list.append(batch['category'])

            if opt.split == 'pred':
                input_img = batch['image'].to(opt.device)
                pred_points = network(input_img, train=False)
                B = pred_points.shape[0]
                gt_points = batch['points'].to(opt.device)
                assert gt_points.shape[0] == B, f'gt {gt_points.shape[0]}, while pred {B}'

                if exp_opts.network == 'atlasnet':
                    pred_points = pred_points.transpose(2, 3).contiguous()
                    pred_points = pred_points.view(B, -1, 3)
                else:
                    pred_points = pred_points

                if exp_opts.mode == "viewer":
                    cam_rotmat = batch['cam_rotmat'].to(opt.device)
                    gt_points = torch.bmm(gt_points, torch.transpose(cam_rotmat, 1, 2))

                pred_loss += eval_loss(gt_points, pred_points).item()
                dist1, dist2, idx1, idx2 = distChamfer(gt_points, pred_points)
                pred_points = pred_points.detach().cpu()
                data_list.append(pred_points)
                opt.type = 'points'

    pred_loss /= len(loader)

    data = torch.cat(data_list, dim=0)
    cats = [item for sublist in cat_list for item in sublist]
    proc_logger.info(f"data shape {data.shape}")


    start_time = time.time()
    if opt.type == 'points':
        data = data.to(opt.device)
        metric = ChamferDistanceL2().to(opt.device)
        #distance_matrix = compute_ptcloud_dismatrix(data, data, metric, proc_logger)
        distance_matrix = compute_ptcloud_dismatrix_batch(data, data, metric, 
                        opt.pred_batch_size, opt.device, proc_logger)
    elif opt.type == 'image':
        data = data.view(data.shape[0], -1)
        try:
            data = data.to(opt.device)
        except:
            data = data
        distance_matrix = pairwise_distances(data)
    else:
        raise NotImplementedError()

    elasp_time = (time.time() - start_time) / 60

    distance_matrix = distance_matrix.cpu().numpy()

    sscore_collect[str(opt.seed)].update({"dm": distance_matrix})
    sscore_collect[str(opt.seed)].update({"class_labels": np.array(cats)})
    sscore_collect[str(opt.seed)].update({"pred_chamfer": pred_loss})


    n_evals = len(opt.perf_pc_list)
    for index in range(n_evals):
        c_method, e_method, n_cluster, perf_pc = opt.c_method[index], opt.e_method[index], opt.cluster_k[index], opt.perf_pc_list[index]

        score, part_label = cluster_eval(c_method=c_method, e_method=e_method, distance_matrix=distance_matrix, 
                seed=opt.seed, n_cluster=n_cluster, pc=perf_pc)

        proc_logger.info(f"{opt.type} mode: {exp_opts.mode}, split: {opt.split} " + 
                    f"nviews: train {opt.nviews_train}, test {opt.nviews_test}, sample num: {sample_num} " + 
                    f"seed{opt.seed}, metric {opt.metric} perf{perf_pc}% " + 
                    f"samp{distance_matrix.shape[0]}, Pred Chamfer: {pred_loss:.6f}, SSCORE: {score:.6f} DM" + 
                    f"{distance_matrix.shape[0]}, compute time {elasp_time:2f} min")

        eval_label = f"{c_method}_{e_method}_k{n_cluster}p{perf_pc}"
        sscore_collect[str(opt.seed)].update({eval_label: {}})
        eval_label_list.add(eval_label)
        sscore_collect[str(opt.seed)][eval_label].update({"sscore": score})
        sscore_collect[str(opt.seed)][eval_label].update({"label": np.array(part_label)})     # cluster label
        sscore_collect[str(opt.seed)][eval_label].update({"perf_percent": perf_pc})
    
eval_label_list = list(eval_label_list)
eval_label_list.sort()
ss_list = {}
for eval_label in eval_label_list:
    ss_list.update({eval_label:[]})

pred_list = []
for seed in sscore_collect:
    pred_list.append(sscore_collect[seed]['pred_chamfer'])
    for eval_label in eval_label_list:
        ss_list[eval_label].append(sscore_collect[seed][eval_label]["sscore"])

for eval_label in eval_label_list:
    ss_mean, ss_std = mean_std(ss_list[eval_label])
    sscore_collect.update({f'{eval_label}': np.array([ss_mean, ss_std])})

pred_loss_mean, pred_loss_std = mean_std(pred_list)

sscore_collect.update({'split': opt.split})
sscore_collect.update({'type': opt.type})
sscore_collect.update({'mode': exp_opts.mode})
sscore_collect.update({'sample_num': sample_num})
sscore_collect.update({'chamfer_stats': np.array([pred_loss_mean, pred_loss_std])})
sscore_collect.update({'trainnv': np.array([opt.nviews_train])})
sscore_collect.update({'testnv': np.array([opt.nviews_test])})

for eval_label in eval_label_list:
    ss_mean, ss_std = sscore_collect[f'{eval_label}'][0], sscore_collect[f'{eval_label}'][1]
    res_logger.info(f"{opt.type} mode: {exp_opts.mode}, split: {opt.split}, " + 
                    f"nviews: train {opt.nviews_train}, test {opt.nviews_test}, testaug{opt.test_augment} sample num: {sample_num} " + 
                    f"seed_list {opt.seed_list}, metric {opt.metric} perf: {perf_pc} % {opt.metric} {opt.trained_exp_dir} {eval_label} " + 
                    f"SSCORE: mean: {ss_mean:.6f}  std: {ss_std:.6f} "+ 
                    f"Pred Chamfer: mean: {pred_loss_mean:.6f}  std: {pred_loss_std:.6f} " +
                    f"DM compute time {elasp_time:.2f} min")
    
np.savez_compressed(os.path.join(res_path, 
f"{exp_opts.mode}_{opt.split}_{opt.type}_testaug{opt.test_augment}_trainnv{opt.nviews_train:02}_testnv{opt.nviews_test:02}_{sample_num}_{opt.trained_exp_dir.split('/')[-1]}.npz"), **sscore_collect)
    
res_logger.info(f"###############END OF {opt.type} {exp_opts.network} {opt.trained_exp_dir} PIPELINE#################")


