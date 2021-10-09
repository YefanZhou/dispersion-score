"""Try out other sklearn score to measure the Train Set points 
"""
import os
import sys
import glob
import numpy as np
import argparse
import tqdm
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from pathlib import Path
sys.path.append('../')
from eval.metric import silhouette, hsic_gam_mat, inertia_ap
from eval.eval_utils import mean_std
from sklearn_extra.cluster import KMedoids
parser = argparse.ArgumentParser()
parser.add_argument("--options", type=int, default=1)
parser.add_argument("--perf_perc", type=int, default=50)
parser.add_argument("--sample_num", type=int, default=20)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--inertia_mean", action='store_true')

args = parser.parse_args()
print(args)
perf_perc = str(args.perf_perc)
#train_gt_path = glob.glob("scripts/shapenet13/pts_traingt_dm_part/*.npz")
train_gt_path = glob.glob("scripts/shapenet13/img_dm_part_sscore/*.npz")
train_gt_path.sort()
#inertia_path = "scripts/shapenet13/pts_traingt_dm_part/inertia_plot"
inertia_path = "scripts/shapenet13/img_dm_part_sscore/inertia_plot"
Path(inertia_path).mkdir(parents=True, exist_ok=True)

# AP + inertia + Pred Nviews=1..23 
# criterion = 'ap_intertia'

# plt.figure(figsize=(10, 10))

# train_nviews = [1, 3, 6, 9, 12, 15, 18, 21, 23]
# pbar = tqdm.tqdm(total=len(train_gt_path) * 5)
# model_inertia_dic = {}
# for i in range(len(train_gt_path)):
#     model_inertia_dic.update({i: []})
# gt_inertia_test = []

# for seed in ['1', '2', '3', '4', '5']: #, 
#     for i in range(len(train_gt_path)):
#         pred_gt = np.load(train_gt_path[i], allow_pickle=True)
#         dm = pred_gt[seed].item()['dm']
#         inertia, matrix_part, part_preference  = inertia_ap(dm, seed=1, pc=args.perf_perc, normalize=False)
#         if args.inertia_mean:
#             model_inertia_dic[i].append(inertia/len(matrix_part))
#         else:
#             model_inertia_dic[i].append(inertia)
#         pbar.update(1)
#     #inertia, matrix_part, part_preference = inertia_ap(tes_gt[str(seed)].item()['dm'], seed=1, pc=args.perf_perc, normalize=False)
#     #gt_inertia_test.append(inertia)
#     #gt_inertia_test.append(kmedoids.inertia_)

    
# val_mean_list = []
# val_std_list = []

# for i in range(len(train_gt_path)):
#     m_s = mean_std(model_inertia_dic[i])
#     val_mean_list.append(m_s[0])
#     val_std_list.append(m_s[1])

# print(val_mean_list)
# print(val_std_list)

# plt.figure(figsize=(10, 10))
# plt.plot(train_nviews, val_mean_list, "-o")
# plt.fill_between(train_nviews, [val_mean_list[i] - val_std_list[i] for i in range(len(val_mean_list))], [val_mean_list[i] + val_std_list[i] for i in range(len(val_mean_list))], facecolor='gray', alpha=0.2)
# #plt.plot(train_nviews, len(train_nviews) * [m_s[0]])
# #plt.fill_between(train_nviews, len(train_nviews) * [m_s[0]-m_s[1]],  len(train_nviews) * [m_s[0]+m_s[1]], facecolor='gray', alpha=0.2)
# plt.xticks(train_nviews)
# plt.legend([f"Train GT {criterion} AP"])
# plt.title(f"Train GT Image {criterion} Value AP Perf {args.perf_perc}")
# plt.xlabel("Num of views per shape in Train Set")
# plt.ylabel(f"{criterion} value")

# if args.inertia_mean:
#     plt.savefig(os.path.join(inertia_path, f'{criterion}_mean_perf{args.perf_perc}.png'))
# else:
#     plt.savefig(os.path.join(inertia_path, f'{criterion}_sum_perf{args.perf_perc}.png'))


criterion = 'KM_inertia_Mean'
n_cluster_list = list(range(10, 91, 10))
#n_cluster_list = list(range(90, 171, 10))
n_cluster = 50
plt.figure(figsize=(15, 10))
for idx, n_cluster in enumerate(n_cluster_list):

    train_nviews = [1, 3, 6, 9, 12, 15, 18, 21, 23]
    pbar = tqdm.tqdm(total=len(train_gt_path) * 5)
    model_inertia_dic = {}
    for i in range(len(train_gt_path)):
        model_inertia_dic.update({i: []})
    gt_inertia_test = []


    for seed in ['1', '2', '3', '4', '5']:
        for i in range(len(train_gt_path)):
            pred_gt = np.load(train_gt_path[i], allow_pickle=True)
            dm = pred_gt[seed].item()['dm']
            kmedoids = KMedoids(n_clusters=n_cluster, random_state=int(seed), metric='precomputed', init='k-medoids++').fit(dm)
            #model_inertia_dic[i].append(kmedoids.inertia_)
            model_inertia_dic[i].append(kmedoids.inertia_ ) #inertia_path
            #model_inertia_dic[i].append(silhouette(dm, kmedoids.labels_))
            
            pbar.update(1)
        #kmedoids = KMedoids(n_clusters=n_cluster, random_state=int(seed), metric='precomputed', init='k-medoids++').fit(tes_gt[str(seed)].item()['dm'])
        #gt_inertia_test.append(kmedoids.inertia_) #/ tes_gt[str(seed)].item()['dm'].shape[0]
        #gt_inertia_test.append(silhouette(tes_gt[str(seed)].item()['dm'], kmedoids.labels_))
         

    val_mean_list = []
    val_std_list = []

    for i in range(len(train_gt_path)):
        m_s = mean_std(model_inertia_dic[i])
        val_mean_list.append(m_s[0])
        val_std_list.append(m_s[1])

    if gt_inertia_test:
        m_s = mean_std(gt_inertia_test)

    #print(val_mean_list)
    #print(val_std_list)
    #print(m_s)
    
    plt.subplot(3, 3, idx // 3 * 3 + idx % 3 + 1)
    plt.plot(train_nviews, val_mean_list, "-o")
    plt.fill_between(train_nviews, [val_mean_list[i] - val_std_list[i] for i in range(len(val_mean_list))], [val_mean_list[i] + val_std_list[i] for i in range(len(val_mean_list))], facecolor='gray', alpha=0.2)
    #plt.plot(train_nviews, len(train_nviews) * [m_s[0]])
    #plt.fill_between(train_nviews, len(train_nviews) * [m_s[0]-m_s[1]],  len(train_nviews) * [m_s[0]+m_s[1]], facecolor='gray', alpha=0.2)
    plt.xticks(train_nviews)
    plt.legend([f"GT {criterion} K={n_cluster}"])
    if idx // 3 == 0:
        plt.title(f"GT Image {criterion} Value")
    if idx // 3 == 2:
        plt.xlabel("Num of views per shape in Train Set")
    if idx % 3 == 0:
        plt.ylabel(f"{criterion} value")

plt.savefig(os.path.join(inertia_path, f'{criterion}_{n_cluster_list[0]}_{n_cluster_list[-1]}.png'))
#plt.show()