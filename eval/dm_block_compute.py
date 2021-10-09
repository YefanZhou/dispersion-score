import os
import torch 
import numpy as np
import tqdm
import sys
import time
# import random
# import time
# import argparse
# import logging
# import pandas as pd 
# from prepare_data import data_prepare
# sys.path.append('../')
# from pathlib import Path
# from os.path import join
# from eval.compute_prediction import silhouette_score
# from dataset.dataset_dm import DmData
# from utils.loss import ChamferDistanceL2, VGGPerceptualLoss, PerceptualLoss

# parser = argparse.ArgumentParser(sys.argv[0])
# parser.add_argument("--data_basedir", type=str, default="../../../../public/zyf/What3D")
# parser.add_argument("--exp_base", type=str, default="../experiment")
# parser.add_argument("--exp_dir", type=str, default="../experiment/recons_extreme/201210_0008_viewer_baseline_trainvws1_seed1")
# parser.add_argument("--exp_argname", type=str, default="args.yaml")
# parser.add_argument("--res_folder", type=str, default="dm_part_sscore_results")
# parser.add_argument("--cache_folder", type=str, default="pts_cache")

# parser.add_argument("--nviews", type=int, default=1)
# parser.add_argument("--mode", type=str, default="viewer")
# parser.add_argument("--split", type=str, default="train")
# parser.add_argument("--type", type=str, default="ptcloud")

# parser.add_argument("--metric", type=str, choices=["chamfer", "mse", "perceptual"])
# parser.add_argument("--seed", type=int, default=1)
# parser.add_argument("--rsample", type=float, default=0.05)    #0.05
# parser.add_argument("--nsample", type=int,  default=-1)       #0.1
# parser.add_argument("--perceptual_batch_size", type=int, default=32)
# parser.add_argument("--num_worker", type=int, default=16)
# parser.add_argument("--content_lr_idx", type=int, default=2)
# parser.add_argument("--perf_pc", type=int,  default=10)
# parser.add_argument("--st_ind_row_ratio", type=float, default=0.75)
# parser.add_argument("--ed_ind_row_ratio", type=float, default=1)


# args = parser.parse_args(sys.argv[1:])
# args.device = torch.device("cuda")


def Block_compute_ptcloud_dismatrix(X1, X2, distance_metric, 
                    st_ind_row, ed_ind_row, logger=None, device=None):

    """return distance matrix between ptcloud X1 ptcloud X2
    Params:
    ----------------------------------
    X1: (N, ptnum, 3) torch.tensor
        point cloud set 1
    X2: (N, ptnum, 3) torch.tensor
        point cloud set 2
    distance_metric: func
        metric to measure the distance of two point cloud
    start_index: int
        index to start the block matrix computing 
    end_index:  int 
        index to end the block matrix computing 
    ifsave: boolean
        if to save the distance matrix to disk
    title: string
        name of the saved matrix
    results_dir: string
        the path to save the distance matrix
    *
    * *
    ----
    * * * 
    * * * *
    --------------------- 
    Returns:
    -------------------------#---------
    D: (ptnum, ptnum) torch.tensor
        distance matrix
    """
    
    N = X1.shape[0]
    # initialize distance matrix
    D = torch.zeros([N, N])
    # iterate over one group of ptcloud
    num_compt = (st_ind_row + ed_ind_row) * (ed_ind_row - st_ind_row) / 2
    pbar = tqdm.tqdm(total=num_compt, desc=f"computing chamfer {st_ind_row}-{ed_ind_row}")
    counter = 0
    report = False
    start_time = time.time()
    with torch.set_grad_enabled(False):
        for row in range(st_ind_row, ed_ind_row):
            for col in range(0, row):
                D[row, col] = distance_metric(X1[row].unsqueeze(0), X2[col].unsqueeze(0))
                pbar.update(1)
                counter += 1 
                if (counter % 20000 == 0 and counter != 0 and not report):
                    elap_min = (time.time() - start_time) / 60 
                    remain_min = (elap_min / counter) * (num_compt - counter)
                    logger.info(f"point cloud {st_ind_row} - {ed_ind_row} Remaining time {remain_min:2f} min")
                    report = True
        pbar.close()
    return D


def compute_ptcloud_dismatrix(X1, X2, distance_metric, logger=None):

    """return distance matrix between ptcloud X1 ptcloud X2
    Params:
    ----------------------------------
    X1: (N, ptnum, 3) torch.tensor
        point cloud set 1
    X2: (N, ptnum, 3) torch.tensor
        point cloud set 2
    distance_metric: func
        metric to measure the distance of two point cloud
    Returns:
    -------------------------#---------
    D: (ptnum, ptnum) torch.tensor
        distance matrix
    """
    
    N = X1.shape[0]
    # initialize distance matrix
    D = torch.zeros([N, N])
    # iterate over one group of ptcloud
    num_compt = N * N / 2
    pbar = tqdm.tqdm(total=num_compt, desc=f"computing....DM")
    counter = 0
    report = False
    start_time = time.time()
    with torch.set_grad_enabled(False):
        for i in range(0, N):
            for j in range(i + 1, N):
                D[i, j] = distance_metric(X1[i].unsqueeze(0), X2[j].unsqueeze(0))
                D[j, i] = D[i, j]
                pbar.update(1)
                counter += 1 
                if (counter % 20000 == 0 and counter != 0 and not report):
                    elap_min = (time.time() - start_time) / 60 
                    remain_min = (elap_min / counter) * (num_compt - counter)
                    logger.info(f"point cloud Remaining time {remain_min:2f} min")
                    report = True
        pbar.close()
        
    return D

# ## Mkdir logger
# res_path = join(args.exp_dir, args.res_folder, f"seed{args.seed}")
# cache_path = join(res_path, args.cache_folder)
# dm_cache_path = join(cache_path, 'dm_part')
# Path(dm_cache_path).mkdir(parents=True, exist_ok=True)

# logger = logging.getLogger()
# file_log_handler = logging.FileHandler(os.path.join(res_path, f"SScore_seed{args.seed}_perf{args.perf_pc}.log"))
# stderr_log_handler = logging.StreamHandler(sys.stdout)
# logger.addHandler(file_log_handler)
# logger.addHandler(stderr_log_handler)
# logger.setLevel("INFO")

# cached_data = data_prepare(args.data_basedir, args.device, rsample=args.rsample, nsample=args.nsample, 
#                     nviews=args.nviews, split_name=args.split, 
#                     mode=args.mode, type=args.type, seed=args.seed, 
#                     perceptual_batch_size=args.perceptual_batch_size, 
#                     content_lr_idx=args.content_lr_idx, logger=logger)

# if args.type != "feats":
#     cached_data = cached_data.to(args.device)

# st_ind_row = int(args.st_ind_row_ratio * cached_data.shape[0])
# ed_ind_row = int(args.ed_ind_row_ratio * cached_data.shape[0])

# ###Metrics 
# if args.metric == "chamfer":
#     metric = ChamferDistanceL2().to(args.device)

# elif args.metric == "mse":   
#     metric = None

# elif args.metric == "perceptual":
#     metric = None

# title = f"{args.mode}_{args.split}_{args.type}_{args.metric}_views{args.nviews}_dm.npy"
# title = f"{args.st_ind_row_ratio:.2f}_{args.ed_ind_row_ratio:.2f}_" + title

# Block_compute_ptcloud_dismatrix(cached_data, cached_data, metric, st_ind_row,
#                          ed_ind_row, logger, title, dm_cache_path, ifsave=True, type=args.type, device=args.device)


