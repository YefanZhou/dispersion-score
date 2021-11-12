"""This script is used to measure output dispersion score of synthetic datasets
"""
import os
import sys
import numpy as np
import torch
import random
import tqdm
import time
from pathlib import Path
from os.path import join
from model.model import EncoderDecoder
sys.path.append(join(os.path.dirname(os.path.abspath(__file__)), "../"))
from dataset.toy_dataset.toydataset import ToyDataset
from auxiliary.my_utils import plant_seeds
from auxiliary.metric_parser import parser
from model.pseudo_network import Generator
from eval.metric import ChamferDistanceL2, compute_ptcloud_dismatrix_batch, cluster_eval
from eval.eval_utils import get_logger, CountFrequency, dic_to_array, mean_std
import auxiliary.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D
from train_toydata import Trainer

opt = parser()
###Mkdir and logger
opt.device = torch.device("cuda")
res_path = join(opt.dir_name, opt.res_folder)
Path(res_path).mkdir(parents=True, exist_ok=True)
proc_logger = get_logger("process", res_path, "process.log")
res_logger = get_logger("results", res_path, "score.log")
opt.logger = proc_logger
print(opt.trained_exp_dir)


nviews_dic = {"train":opt.nviews_train, "test":opt.nviews_test}
num_seed = max(len(opt.seed_list), 1)
score_collect = {}
eval_label_list = set()

for seed_idx in range(num_seed):
    if opt.seed_list:
        opt.seed = opt.seed_list[seed_idx]
    score_collect.update({str(opt.seed):{}})
    plant_seeds(opt.seed)

    ##Loading Data and Network
    if opt.split == 'pred':
        eval_loss = ChamferDistanceL2().to(opt.device)
        distChamfer = dist_chamfer_3D.chamfer_3DDist()

        if opt.network=='atlasnet':
            network = EncoderDecoder(opt)
            opt.logger.info(f"Reloading Network Weights from {opt.reload_model_path}...")
            network.load_state_dict(torch.load(opt.reload_model_path)['model_state_dict'])
            network.to(opt.device)

        # if opt.network=='pseudo_network':
        #     proc_logger.info(f"Network {opt.network}: From {os.path.join(opt.trained_exp_dir, 'prediction.npy')}")
        #     data = np.load(os.path.join(opt.trained_exp_dir, 'prediction.npy'))
        #     data = torch.from_numpy(data).to(opt.device)
        #     network = Generator(data, opt.pred_batch_size)
        # else:
        #     raise NotImplementedError(f"{opt.network} is not implemented/imported")
    
    if opt.split == "train":
        dataset = ToyDataset(data_base_dir=opt.data_base_dir, 
                            json_file=opt.train_json_file,
                            num_points=opt.number_points, 
                            train=True, 
                            normalization=opt.normalization, 
                            logger=opt.logger) 
    elif opt.split == "test" or opt.split == "pred":
        dataset = ToyDataset(data_base_dir=opt.data_base_dir, 
                            json_file=opt.test_json_file,
                            num_points=opt.number_points, 
                            train=False, 
                            normalization=opt.normalization, 
                            logger=opt.logger) 
    else:
        raise NotImplementedError()

    loader = torch.utils.data.DataLoader(dataset, 
                                        batch_size=opt.pred_batch_size, 
                                        shuffle=False, num_workers=8)
    if opt.rsample == 1:
        sample_num = len(dataset)
        opt.nsample = len(dataset)
    else:
        if opt.rsample != -1:
            opt.nsample = int(opt.rsample * len(dataset))
        subset_index = random.sample(range(len(dataset)), opt.nsample)
        dataset = torch.utils.data.Subset(dataset, subset_index)
        sample_num = len(subset_index)
    data = None
    pred_loss = 0.0
    with torch.set_grad_enabled(False): 
        for batch in tqdm.tqdm(loader, desc=f"loading {opt.split} {opt.type} data"):
            if opt.split == 'pred':
                input_img = batch['image'].to(opt.device)
                pred_points = network(input_img, train=False)
                pred_points = pred_points.transpose(2, 3).contiguous()
                B = pred_points.shape[0]
                pred_points = pred_points.view(B, -1, 3)
                gt_points = batch['points'].to(opt.device)
                assert gt_points.shape[0] == B, f'gt {gt_points.shape[0]}, while pred {B}'
                print(gt_points.shape)
                print(pred_points.shape)
                if data is None:
                    data = pred_points
                else:
                    data = torch.cat((data, pred_points), dim=0)
                pred_loss += eval_loss(gt_points, pred_points).item()
                dist1, dist2, idx1, idx2 = distChamfer(gt_points, pred_points)
                opt.type = 'points'

    pred_loss /= len(loader)
    proc_logger.info(f"Pred Chamfer Loss: {pred_loss:4f}")
    start_time = time.time()

    if opt.type == 'points':
        data = data.to(opt.device)
        metric = ChamferDistanceL2().to(opt.device)
        distance_matrix = compute_ptcloud_dismatrix_batch(data, data, metric, 
                        opt.pred_batch_size, opt.device, proc_logger)
    else:
        raise NotImplementedError()

    elasp_time = (time.time() - start_time) / 60

    distance_matrix = distance_matrix.cpu().numpy()

    score_collect[str(opt.seed)].update({"dm": distance_matrix})
    score_collect[str(opt.seed)].update({"pred_chamfer": pred_loss})
    
    n_evals = len(opt.perf_pc_list)
    for index in range(n_evals):
        c_method, e_method, n_cluster, perf_pc = opt.c_method[index], opt.e_method[index], opt.cluster_k[index], opt.perf_pc_list[index]

        score, part_label = cluster_eval(c_method=c_method, e_method=e_method, distance_matrix=distance_matrix, 
                seed=opt.seed, n_cluster=n_cluster, pc=perf_pc)

        label_stat_verbose = ""
        freq = CountFrequency(part_label)
        for key, value in freq.items(): 
            label_stat_verbose += "% d :% d | "%(key, value)

        proc_logger.info(f"{opt.type} mode: {opt.mode}, split: {opt.split} " + 
                    f"nviews: train {opt.nviews_train}, test {opt.nviews_test}, sample num:{sample_num} " + 
                    f"seed{opt.seed}, metric{opt.metric} perf{perf_pc}% " + 
                    f"samp{distance_matrix.shape[0]}, Pred Chamfer: {pred_loss:.4f}, score: {score:.4f} DM" + 
                    f"{distance_matrix.shape[0]}, compute time {elasp_time:2f} min")

        eval_label = f"{c_method}_{e_method}_k{n_cluster}p{perf_pc}"
        score_collect[str(opt.seed)].update({eval_label: {}})
        eval_label_list.add(eval_label)
        score_collect[str(opt.seed)][eval_label].update({"score": score})
        score_collect[str(opt.seed)][eval_label].update({"label": np.array(part_label)})     # cluster label
        score_collect[str(opt.seed)][eval_label].update({"perf_percent": perf_pc})
        score_collect[str(opt.seed)][eval_label].update({"label_stats": dic_to_array(freq)})
    
eval_label_list = list(eval_label_list)
eval_label_list.sort()
ss_list = {}
for eval_label in eval_label_list:
    ss_list.update({eval_label:[]})

pred_list = []

for seed in score_collect:
    pred_list.append(score_collect[seed]['pred_chamfer'])
    for eval_label in eval_label_list:
        ss_list[eval_label].append(score_collect[seed][eval_label]["score"])

for eval_label in eval_label_list:
    avg_score_lst = [score/sample_num for score in ss_list[eval_label]]
    ss_mean, ss_std = mean_std(ss_list[eval_label])
    avg_ss_mean, avg_ss_std = mean_std(avg_score_lst)
    score_collect.update({f'{eval_label}': np.array([ss_mean, ss_std])})
    score_collect.update({f'avg_{eval_label}': np.array([avg_ss_mean, avg_ss_std])})

pred_loss_mean, pred_loss_std = mean_std(pred_list)

score_collect.update({'split': opt.split})
score_collect.update({'type': opt.type})
score_collect.update({'mode': opt.mode})
score_collect.update({'sample_num': sample_num})
score_collect.update({'chamfer_stats': np.array([pred_loss_mean, pred_loss_std])})
score_collect.update({'trainnv': np.array([opt.nviews_train])})
score_collect.update({'testnv': np.array([opt.nviews_test])})

for eval_label in eval_label_list:
    ss_mean, ss_std = score_collect[f'{eval_label}'][0], score_collect[f'{eval_label}'][1]
    avg_ss_mean, avg_ss_std = score_collect[f'avg_{eval_label}'][0], score_collect[f'avg_{eval_label}'][1]
    res_logger.info(f"{opt.network} {opt.type} mode: {opt.mode}, split: {opt.split}, " + 
                    f"nviews: train {opt.nviews_train}, test {opt.nviews_test}, sample num: {sample_num} " + 
                    f"seed_list {opt.seed_list}, metric {opt.metric} perf: {perf_pc} % {opt.metric} {opt.trained_exp_dir} {eval_label} " + 
                    f"Sum of Score: (mean: {ss_mean:.4f}|std: {ss_std:.4f})  "+ 
                    f"Average Score: (mean: {avg_ss_mean:.4f}|std: {avg_ss_std:.4f})   "+ 
                    f"Pred Chamfer: (mean:{pred_loss_mean:.4f}|std: {pred_loss_std:.4f})  " +
                    f"DM compute time {elasp_time:.2f} min")
    
np.savez_compressed(os.path.join(res_path, 
f"{opt.network}_{opt.mode}_{opt.split}_{opt.type}_{sample_num}_{opt.trained_exp_dir.split('/')[-1]}.npz"), **score_collect)
    
res_logger.info(f"###############END OF {opt.type} {opt.network} {opt.trained_exp_dir} PIPELINE#################")



