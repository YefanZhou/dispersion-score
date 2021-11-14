import os
import numpy as np
import sys
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import random
import tqdm
import time
from pathlib import Path
from os.path import join
sys.path.append(join(os.path.dirname(os.path.abspath(__file__)), "../"))
import dataset.dataset_shapenet_views_yawlimit as dataset_shapenet_views_yawlimit
from auxiliary.my_utils import plant_seeds, chunks
from auxiliary.metric_parser import parser
from eval.metric import ChamferDistanceL2, PerceptualEncoder, cluster_eval, compute_ptcloud_dismatrix_batch, compute_img_dismatrix_batch
from eval.eval_utils import get_logger, mean_std

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

yawrange = opt.rendering_root_dir.split('/')[-2]

for seed_idx in range(num_seed):
    if opt.seed_list:
        opt.seed = opt.seed_list[seed_idx]
    sscore_collect.update({str(opt.seed):{}})
    plant_seeds(opt.seed)
    ## train
    if opt.split == "train":
        dataset = dataset_shapenet_views_yawlimit.ShapeNet(opt, train=True, num_image_per_object=nviews_dic[opt.split]) 
    elif opt.split == "test":
        dataset = dataset_shapenet_views_yawlimit.ShapeNet(opt, train=False, num_image_per_object=nviews_dic[opt.split]) 
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

    for batch in tqdm.tqdm(loader, desc=f"loading {opt.split} {opt.type} {opt.mode} data"):
        if opt.type == "points" and opt.mode == "viewer":
            cam_rotmat = batch['cam_rotmat']
            points = torch.bmm(batch[opt.type], torch.transpose(cam_rotmat, 1, 2))
            data_list.append(points)
        else:
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
    np.save(os.path.join(res_path, f"anglelimit_yawrange{yawrange}_{opt.mode}_{opt.split}_{opt.type}_trainnv{opt.nviews_train:02}_testnv{opt.nviews_test:02}_{sample_num}_SEED{opt.seed}.npy"), distance_matrix)
    
    n_evals = len(opt.perf_pc_list)
    for index in range(n_evals):
        c_method, e_method, n_cluster, perf_pc = opt.c_method[index], opt.e_method[index], opt.cluster_k[index], opt.perf_pc_list[index]

        score, part_label = cluster_eval(c_method=c_method, e_method=e_method, distance_matrix=distance_matrix, 
                seed=opt.seed, n_cluster=n_cluster, pc=perf_pc)

        proc_logger.info(f"{opt.type} mode: {opt.mode}, split: {opt.split} " + 
                    f"nviews: train {opt.nviews_train}, test {opt.nviews_test}, sample num: {sample_num} " + 
                    f"seed{opt.seed}, metric {opt.metric}, {opt.img_aug_type} {opt.autoaug_type} mag{opt.mag_idx} n_op{opt.n_op} prob{opt.prob} {c_method}, {e_method}, cluster_k {n_cluster},  perf{perf_pc}% " + 
                    f"samp{distance_matrix.shape[0]}, SCORE: {score:.6f} DM" + 
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

for seed in sscore_collect:
    for eval_label in eval_label_list:
        ss_list[eval_label].append(sscore_collect[seed][eval_label]["sscore"])

for eval_label in eval_label_list:
    ss_mean, ss_std = mean_std(ss_list[eval_label])
    sscore_collect.update({f'{eval_label}': np.array([ss_mean, ss_std])})

sscore_collect.update({'split': opt.split})
sscore_collect.update({'type': opt.type})
sscore_collect.update({'mode': opt.mode})
sscore_collect.update({'sample_num': sample_num})
sscore_collect.update({'trainnv': np.array([opt.nviews_train])})
sscore_collect.update({'testnv': np.array([opt.nviews_test])})


for eval_label in eval_label_list:
    ss_mean, ss_std = sscore_collect[f'{eval_label}'][0], sscore_collect[f'{eval_label}'][1]
    res_logger.info(f"{opt.type} mode: {opt.mode}, split: {opt.split}, " + 
                    f"nviews: train {opt.nviews_train}, test {opt.nviews_test}, anglelimit_yawrange {yawrange}, sample num: {sample_num} " + 
                    f"seed_list {opt.seed_list}, metric {opt.metric} {opt.img_aug_type} {opt.autoaug_type} mag{opt.mag_idx} magnitude {opt.magnitude} n_op{opt.n_op} prob{opt.prob} {eval_label} " + 
                    f"SCORE: mean: {ss_mean:.6f}  std: {ss_std:.6f} "+ 
                    f"DM compute time {elasp_time:.2f} min")
    
np.savez_compressed(os.path.join(res_path, f"anglelimit_yawrange{yawrange}_{opt.mode}_{opt.split}_{opt.type}_{opt.img_aug_type}_{opt.autoaug_type}_mag{opt.mag_idx}_magnitude{opt.magnitude}_n_op{opt.n_op}_prob{opt.prob}_trainnv{opt.nviews_train:02}_testnv{opt.nviews_test:02}_{sample_num}.npz"), **sscore_collect)
    
res_logger.info(f"###############END OF {opt.type} PIPELINE#################")





