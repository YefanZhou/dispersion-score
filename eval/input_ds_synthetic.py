"""This script is used to compute input dispersion score of synthetic dataset
"""
import os
import numpy as np
import sys
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import random
import tqdm
import time
import logging
from easydict import EasyDict
from pathlib import Path
from os.path import join
import torch.nn as nn
sys.path.append(join(os.path.dirname(os.path.abspath(__file__)), "../"))
from dataset.toy_dataset.toydataset import ToyDataset
from dataset.dataset_shapenet_views import ShapeNet
from auxiliary.my_utils import plant_seeds, chunks
from auxiliary.metric_parser import parser
from eval.metric import ChamferDistanceL2, PerceptualEncoder, cluster_eval, compute_ptcloud_dismatrix, compute_ptcloud_dismatrix_batch, silhouette_score_ap, pairwise_distances_torch, compute_img_dismatrix_batch
from eval.eval_utils import get_logger, CountFrequency, dic_to_array, mean_std


opt = parser()
opt.device = torch.device("cuda")
res_path = join(opt.dir_name, opt.res_folder)
Path(res_path).mkdir(parents=True, exist_ok=True)
proc_logger = get_logger("process", res_path, "process.log")
res_logger = get_logger("results", res_path, "score.log")
opt.logger = proc_logger

nviews_dic = {"train":opt.nviews_train, "test":opt.nviews_test}
num_seed = max(len(opt.seed_list), 1)
sscore_collect = {}
eval_label_list = set()
proc_logger.info(opt.c_method)
proc_logger.info(opt.e_method)
proc_logger.info(opt.cluster_k)
proc_logger.info(opt.perf_pc_list)
percep_encoder = PerceptualEncoder().to(opt.device)

for seed_idx in range(num_seed):
    if opt.seed_list:
        opt.seed = opt.seed_list[seed_idx]
    sscore_collect.update({str(opt.seed):{}})

    plant_seeds(opt.seed)
    ## train
    if opt.split == "train":
        #dataset = ShapeNet(opt, train=True, num_image_per_object=nviews_dic[opt.split]) 
        dataset = ToyDataset(opt.data_base_dir, 
                json_file=opt.train_json_file, 
                num_points=opt.number_points, 
                train=True,
                normalization=opt.normalization, 
                logger=opt.logger)

    elif opt.split == "test":
        dataset = ToyDataset(opt.data_base_dir, 
                    json_file=opt.test_json_file, 
                    num_points=opt.number_points, 
                    train=False,
                    normalization=opt.normalization, 
                    logger=opt.logger)
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
        
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    data_list = []

    for batch in tqdm.tqdm(loader, desc=f"loading {opt.split} {opt.type} data"):
        data_list.append(batch[opt.type])

    data = torch.cat(data_list, dim=0)

 
    ## All Perceptual Layer
    if opt.type == "image" and opt.perceptual:
        if opt.perceptual_max_pooling:
            cat_percep_data = torch.zeros(data.shape[0], 56 * 56).to(opt.device)
            proc_logger.info(f"{cat_percep_data.shape} move to device")
            pbar = tqdm.tqdm(total=int(data.shape[0]/opt.perceptual_batch_size), desc='perceptual')
            for batch_image, indexes in zip(data.split(opt.perceptual_batch_size), 
                                            chunks(range(data.shape[0]), opt.perceptual_batch_size)):
                
                batch_image = batch_image.to(opt.device)
                _, _, h_relu_3_3, _ = percep_encoder(batch_image)
                ## Max Pooling 
                h_relu_3_3, _ = torch.max(h_relu_3_3, dim=1)
                h_relu_3_3 = h_relu_3_3.detach().view(h_relu_3_3.shape[0], -1)
                cat_percep_data[indexes[0] : indexes[0] + h_relu_3_3.shape[0]] = h_relu_3_3
                pbar.update(1)
            data = cat_percep_data
        else:
            cat_percep_data = torch.zeros(data.shape[0], 256 * 56 * 56)
            proc_logger.info(f"{cat_percep_data.shape} move to device")
            pbar = tqdm.tqdm(total=int(data.shape[0]/opt.perceptual_batch_size), desc='perceptual')
            for batch_image, indexes in zip(data.split(opt.perceptual_batch_size), 
                                            chunks(range(data.shape[0]), opt.perceptual_batch_size)):
                
                batch_image = batch_image.to(opt.device)
                _, _, h_relu_3_3, _ = percep_encoder(batch_image)
                h_relu_3_3 = h_relu_3_3.detach().cpu().view(h_relu_3_3.shape[0], -1)
                cat_percep_data[indexes[0] : indexes[0] + h_relu_3_3.shape[0]] = h_relu_3_3
                pbar.update(1)
            data = cat_percep_data

    proc_logger.info(f"data shape {data.shape}")

    start_time = time.time()
    if opt.type == 'points':
        data = data.to(opt.device)
        metric = ChamferDistanceL2().to(opt.device)
        distance_matrix = compute_ptcloud_dismatrix_batch(data, data, metric, 
                        opt.pred_batch_size, opt.device, proc_logger)
        del data
        torch.cuda.empty_cache()
    elif opt.type == 'image':
        data = data.view(data.shape[0], -1)
        proc_logger.info(f"feature shape: {data.shape}")
        #data = data.to(opt.device)
        # try:
        #     data = data.to(opt.device)
        # except:
        #     data = data 

        #distance_matrix = pairwise_distances_torch(data)
        metric = torch.nn.MSELoss(reduction='none').to(opt.device)
        distance_matrix = compute_img_dismatrix_batch(data, data, metric, 
                        opt.pred_batch_size, opt.device, proc_logger)
        del data
        del cat_percep_data
        torch.cuda.empty_cache()
    else:
        raise NotImplementedError()

    elasp_time = (time.time() - start_time) / 60

    distance_matrix = distance_matrix.detach().cpu().numpy()
    np.save(os.path.join(res_path, f"dismat_{opt.mode}_{opt.split}_{opt.type}_{opt.train_json_file.split('.')[0]}_numsample{sample_num}.npy"), distance_matrix)

    n_evals = len(opt.perf_pc_list)
    for index in range(n_evals):
        c_method, e_method, n_cluster, perf_pc = opt.c_method[index], opt.e_method[index], opt.cluster_k[index], opt.perf_pc_list[index]

        score, part_label = cluster_eval(c_method=c_method, e_method=e_method, distance_matrix=distance_matrix, 
                seed=opt.seed, n_cluster=n_cluster, pc=perf_pc)

        label_stat_verbose = ""
        freq = CountFrequency(part_label)
        for key, value in freq.items(): 
            label_stat_verbose += "% d :% d | "%(key, value)

        proc_logger.info(f"Type:{opt.type}, mode:{opt.mode}, split:{opt.split}, Subtrainset:{opt.train_json_file}   "+ 
                    f"sample num:{sample_num}   " + 
                    f"seed:{opt.seed}, metric:{opt.metric}, {c_method}, {e_method}, cluster_k:{n_cluster}   " + 
                    f"dismat_size:{distance_matrix.shape[0]}, SCORE:{score:.6f}    " + 
                    f"compute_time:{elasp_time:2f} min")

        eval_label = f"{c_method}_{e_method}_k{n_cluster}p{perf_pc}"
        sscore_collect[str(opt.seed)].update({eval_label: {}})
        eval_label_list.add(eval_label)
        sscore_collect[str(opt.seed)][eval_label].update({"sscore": score})
        sscore_collect[str(opt.seed)][eval_label].update({"avg_sscore": score/sample_num})
        sscore_collect[str(opt.seed)][eval_label].update({"label": np.array(part_label)})     # cluster label
        sscore_collect[str(opt.seed)][eval_label].update({"perf_percent": perf_pc})
        sscore_collect[str(opt.seed)][eval_label].update({"label_stats": dic_to_array(freq)})


eval_label_list = list(eval_label_list)
eval_label_list.sort()

ss_list = {}
for eval_label in eval_label_list:
    ss_list.update({eval_label:[]})

for seed in sscore_collect:
    for eval_label in eval_label_list:
        ss_list[eval_label].append(sscore_collect[seed][eval_label]["sscore"])

for eval_label in eval_label_list:
    avg_score_lst = [score/sample_num for score in ss_list[eval_label]]
    ss_mean, ss_std = mean_std(ss_list[eval_label])
    avg_ss_mean, avg_ss_std = mean_std(avg_score_lst)
    sscore_collect.update({f'{eval_label}': np.array([ss_mean, ss_std])})
    sscore_collect.update({f'avg_{eval_label}': np.array([avg_ss_mean, avg_ss_std])})

sscore_collect.update({'split': opt.split})
sscore_collect.update({'type': opt.type})
sscore_collect.update({'mode': opt.mode})
sscore_collect.update({'sample_num': sample_num})
sscore_collect.update({'trainnv': np.array([opt.nviews_train])})
sscore_collect.update({'testnv': np.array([opt.nviews_test])})


for eval_label in eval_label_list:
    ss_mean, ss_std = sscore_collect[f'{eval_label}'][0], sscore_collect[f'{eval_label}'][1]
    avg_ss_mean, avg_ss_std = sscore_collect[f'avg_{eval_label}'][0], sscore_collect[f'avg_{eval_label}'][1]
    res_logger.info(f"Type:{opt.type}, mode:{opt.mode}, split:{opt.split}, Subtrainset:{opt.train_json_file}  " + 
                    f"sample_num:{sample_num}, seed_list:{opt.seed_list}, metric:{opt.metric}, eval_method:{eval_label}   " + 
                    f"Sum_of_Score: (mean: {ss_mean:.6f}|std: {ss_std:.6f})   "+ 
                    f"Dispersion Score: (mean: {avg_ss_mean:.6f}|std: {avg_ss_std:.6f})  "+ 
                    f"DM compute time {elasp_time:.2f} min")
    
np.savez_compressed(os.path.join(res_path, f"stats_{opt.mode}_{opt.split}_{opt.type}_{opt.train_json_file.split('.')[0]}_numsample{sample_num}.npz"), **sscore_collect)
res_logger.info(f"###############END OF {opt.type} PIPELINE#################")





