"""Try out other sklearn score to measure the 
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
from eval.metric import silhouette, hsic_gam_mat, inertia_ap, DaviesBouldin
from eval.eval_utils import mean_std
from sklearn_extra.cluster import KMedoids
parser = argparse.ArgumentParser()
parser.add_argument("--options", type=int, default=1)
parser.add_argument("--perf_perc", type=int, default=50)
parser.add_argument("--sample_num", type=int, default=20)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--inertia_mean", action='store_true')
parser.add_argument("--normalize", action='store_true')


args = parser.parse_args()
print(args)
perf_perc = str(args.perf_perc)
test_gt_path = 'scripts/shapenet13/pts_testgt_dm_part_sscore/viewer_test_points_trainnv01_testnv01_2000.npz'
pred_path = glob.glob("scripts/shapenet13_pred/dm_part_sscore/*.npz")
pred_path.sort()
pred_path.remove('scripts/shapenet13_pred/dm_part_sscore/viewer_pred_points_trainnv02_testnv01_2000.npz')
pred_path.remove('scripts/shapenet13_pred/dm_part_sscore/viewer_pred_points_trainnv04_testnv01_2000.npz')
pred_path.remove('scripts/shapenet13_pred/dm_part_sscore/viewer_pred_points_trainnv05_testnv01_2000.npz')

tes_gt = np.load(test_gt_path, allow_pickle=True)

model_outlier_dic = {}
for i in range(len(pred_path)):
    model_outlier_dic.update({i: []})

gt_outlier_test = []

## LOF, fit pred and predict
# for seed in ['1', '2', '3', '4', '5']:
#     for i in range(len(pred_path)):
#         pred_gt = np.load(pred_path[i], allow_pickle=True)
#         dm = pred_gt[seed].item()['dm']
#         clf = LocalOutlierFactor(metric='precomputed')
#         clf.fit_predict(dm)
#         model_outlier_dic[i].append(-1 * np.mean(clf.negative_outlier_factor_))
#         #print(len(model_outlier_dic[i]))
#     clf = LocalOutlierFactor(metric='precomputed')
#     clf.fit_predict(tes_gt[seed].item()['dm'])
#     gt_outlier_test.append(-1 * np.mean(clf.negative_outlier_factor_))
    
## LOF, fit gt and predict
# for seed in ['1', '2', '3', '4', '5']:
#     clf = LocalOutlierFactor(metric='precomputed', novelty=True)
#     clf.fit(tes_gt[seed].item()['dm'])
#     for i in range(len(pred_path)):
#         pred_gt = np.load(pred_path[i], allow_pickle=True)
#         dm = pred_gt[seed].item()['dm']
#         model_outlier_dic[i].append(-1 * np.mean(clf.score_samples(dm)))
        
#     #gt_outlier_test.append(-1 * np.mean(clf.negative_outlier_factor_))

# for i in range(len(pred_path)):
#     print(pred_path[i])
#     m_s = mean_std(model_outlier_dic[i])
#     print(f"{m_s[0]:.6f}, {m_s[1]:.6f}")

# if gt_outlier_test:
#     m_s = mean_std(gt_outlier_test)
#     print(f"{m_s[0]:.6f}, {m_s[1]:.6f}")


# # k = 50      90
## Kmeans + inertia 
criterion = 'KM_inertia_Mean'
n_cluster_list = list(range(10, 91, 10))
#n_cluster_list = list(range(90, 171, 10))
n_cluster = 50
plt.figure(figsize=(10, 10))
for idx, n_cluster in enumerate(n_cluster_list):

    train_nviews = [1, 3, 6, 9, 12, 15, 18, 21, 23]
    pbar = tqdm.tqdm(total=len(pred_path) * 5)
    model_inertia_dic = {}
    for i in range(len(pred_path)):
        model_inertia_dic.update({i: []})
    gt_inertia_test = []


    for seed in ['1', '2', '3', '4', '5']:
        for i in range(len(pred_path)):
            pred_gt = np.load(pred_path[i], allow_pickle=True)
            dm = pred_gt[seed].item()['dm']
            kmedoids = KMedoids(n_clusters=n_cluster, random_state=int(seed), metric='precomputed', init='k-medoids++').fit(dm)
            #model_inertia_dic[i].append(kmedoids.inertia_)
            model_inertia_dic[i].append(kmedoids.inertia_ / dm.shape[0])
            #model_inertia_dic[i].append(silhouette(dm, kmedoids.labels_))
            
            pbar.update(1)
        kmedoids = KMedoids(n_clusters=n_cluster, random_state=int(seed), metric='precomputed', init='k-medoids++').fit(tes_gt[str(seed)].item()['dm'])
        gt_inertia_test.append(kmedoids.inertia_ / tes_gt[str(seed)].item()['dm'].shape[0])
        #gt_inertia_test.append(silhouette(tes_gt[str(seed)].item()['dm'], kmedoids.labels_))
         

    val_mean_list = []
    val_std_list = []

    for i in range(len(pred_path)):
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
    plt.plot(train_nviews, len(train_nviews) * [m_s[0]])
    plt.fill_between(train_nviews, len(train_nviews) * [m_s[0]-m_s[1]],  len(train_nviews) * [m_s[0]+m_s[1]], facecolor='gray', alpha=0.2)
    plt.xticks(train_nviews)
    plt.legend([f"Pred {criterion} K={n_cluster}", f"GT {criterion} K={n_cluster}"])
    if idx // 3 == 0:
        plt.title(f"Pred PointCloud {criterion} Value")
    if idx // 3 == 2:
        plt.xlabel("Num of views per shape in Train Set")
    if idx % 3 == 0:
        plt.ylabel(f"{criterion} value")

plt.savefig(f'scripts/shapenet13_pred/{criterion}_{n_cluster_list[0]}_{n_cluster_list[-1]}.png')

# criterion = 'KM_inertia_Mean'
# n_cluster_list = list(range(10, 91, 10))
# #n_cluster_list = list(range(90, 171, 10))
# n_cluster = 50
# plt.figure(figsize=(10, 10))
# for idx, n_cluster in enumerate(n_cluster_list):

#     train_nviews = [1, 3, 6, 9, 12, 15, 18, 21, 23]
#     pbar = tqdm.tqdm(total=len(pred_path) * 5)
#     model_inertia_dic = {}
#     for i in range(len(pred_path)):
#         model_inertia_dic.update({i: []})
#     gt_inertia_test = []


#     for seed in ['1', '2', '3', '4', '5']:
#         for i in range(len(pred_path)):
#             pred_gt = np.load(pred_path[i], allow_pickle=True)
#             dm = pred_gt[seed].item()['dm']
#             kmedoids = KMedoids(n_clusters=n_cluster, random_state=int(seed), metric='precomputed', init='k-medoids++').fit(dm)
#             #model_inertia_dic[i].append(kmedoids.inertia_)
#             model_inertia_dic[i].append(kmedoids.inertia_ / dm.shape[0])
#             #model_inertia_dic[i].append(silhouette(dm, kmedoids.labels_))
            
#             pbar.update(1)
#         kmedoids = KMedoids(n_clusters=n_cluster, random_state=int(seed), metric='precomputed', init='k-medoids++').fit(tes_gt[str(seed)].item()['dm'])
#         gt_inertia_test.append(kmedoids.inertia_ / tes_gt[str(seed)].item()['dm'].shape[0])
#         #gt_inertia_test.append(silhouette(tes_gt[str(seed)].item()['dm'], kmedoids.labels_))
         

#     val_mean_list = []
#     val_std_list = []

#     for i in range(len(pred_path)):
#         m_s = mean_std(model_inertia_dic[i])
#         val_mean_list.append(m_s[0])
#         val_std_list.append(m_s[1])

#     if gt_inertia_test:
#         m_s = mean_std(gt_inertia_test)

#     #print(val_mean_list)
#     #print(val_std_list)
#     #print(m_s)
    
#     plt.subplot(3, 3, idx // 3 * 3 + idx % 3 + 1)
#     plt.plot(train_nviews, val_mean_list, "-o")
#     plt.fill_between(train_nviews, [val_mean_list[i] - val_std_list[i] for i in range(len(val_mean_list))], [val_mean_list[i] + val_std_list[i] for i in range(len(val_mean_list))], facecolor='gray', alpha=0.2)
#     plt.plot(train_nviews, len(train_nviews) * [m_s[0]])
#     plt.fill_between(train_nviews, len(train_nviews) * [m_s[0]-m_s[1]],  len(train_nviews) * [m_s[0]+m_s[1]], facecolor='gray', alpha=0.2)
#     plt.xticks(train_nviews)
#     plt.legend([f"Pred {criterion} K={n_cluster}", f"GT {criterion} K={n_cluster}"])
#     if idx // 3 == 0:
#         plt.title(f"Pred PointCloud {criterion} Value")
#     if idx // 3 == 2:
#         plt.xlabel("Num of views per shape in Train Set")
#     if idx % 3 == 0:
#         plt.ylabel(f"{criterion} value")

# plt.savefig(f'scripts/shapenet13_pred/{criterion}_{n_cluster_list[0]}_{n_cluster_list[-1]}.png')


## AP + inertia + Pred Nviews=1..23 
# criterion = 'ap_inertia_normalize'

# plt.figure(figsize=(10, 10))

# train_nviews = [1, 3, 6, 9, 12, 15, 18, 21, 23]   #
# pbar = tqdm.tqdm(total=len(pred_path) * 5)
# model_inertia_dic = {}
# for i in range(len(pred_path)):
#     model_inertia_dic.update({i: []})
# gt_inertia_test = []

# for seed in ['1', '2', '3', '4', '5']:
#     for i in range(len(pred_path)):
#         pred_gt = np.load(pred_path[i], allow_pickle=True)
#         dm = pred_gt[seed].item()['dm']
#         inertia, matrix_part, part_preference  = inertia_ap(dm, seed=1, pc=args.perf_perc, normalize=True)
#         model_inertia_dic[i].append(inertia)
#         pbar.update(1)
#     inertia, matrix_part, part_preference = inertia_ap(tes_gt[str(seed)].item()['dm'], seed=1, pc=args.perf_perc, normalize=True)
#     gt_inertia_test.append(inertia)
#     #gt_inertia_test.append(kmedoids.inertia_)
#     #gt_inertia_test.append(silhouette(tes_gt[str(seed)].item()['dm'], kmedoids.labels_))
    
# val_mean_list = []
# val_std_list = []

# for i in range(len(pred_path)):
#     m_s = mean_std(model_inertia_dic[i])
#     val_mean_list.append(m_s[0])
#     val_std_list.append(m_s[1])

# if gt_inertia_test:
#     m_s = mean_std(gt_inertia_test)

# print(val_mean_list)
# print(val_std_list)
# print(m_s)

# plt.figure(figsize=(10, 10))
# plt.plot(train_nviews, val_mean_list, "-o")
# plt.fill_between(train_nviews, [val_mean_list[i] - val_std_list[i] for i in range(len(val_mean_list))], [val_mean_list[i] + val_std_list[i] for i in range(len(val_mean_list))], facecolor='gray', alpha=0.2)
# plt.plot(train_nviews, len(train_nviews) * [m_s[0]])
# plt.fill_between(train_nviews, len(train_nviews) * [m_s[0]-m_s[1]],  len(train_nviews) * [m_s[0]+m_s[1]], facecolor='gray', alpha=0.2)
# plt.xticks(train_nviews)
# plt.legend([f"Pred {criterion} AP", f"GT {criterion} AP"])
# plt.title(f"Pred PointCloud {criterion} Value AP Perf {args.perf_perc}")
# plt.xlabel("Num of views per shape in Train Set")
# plt.ylabel(f"{criterion} value")
# plt.savefig(f'scripts/shapenet13_pred/{criterion}_perf{args.perf_perc}.png')
#plt.show()


# k = 50      90
## Kmeans + DBI 
# criterion = 'DBI_mean'
# #n_cluster_list = list(range(10, 91, 10))
# n_cluster_list = list(range(90, 171, 10))
# n_cluster = 50
# plt.figure(figsize=(10, 10))
# for idx, n_cluster in enumerate(n_cluster_list):

#     train_nviews = [1, 3, 6, 9, 12, 15, 18, 21, 23]
#     pbar = tqdm.tqdm(total=len(pred_path) * 5)
#     model_inertia_dic = {}
#     for i in range(len(pred_path)):
#         model_inertia_dic.update({i: []})
#     gt_inertia_test = []


#     for seed in ['1', '2', '3', '4', '5']:
#         for i in range(len(pred_path)):
#             pred_gt = np.load(pred_path[i], allow_pickle=True)
#             dm = pred_gt[seed].item()['dm']
#             kmedoids = KMedoids(n_clusters=n_cluster, random_state=int(seed), metric='precomputed', init='k-medoids++').fit(dm)
#             #model_inertia_dic[i].append(kmedoids.inertia_)
#             #model_inertia_dic[i].append(silhouette(dm, kmedoids.labels_))
#             model_inertia_dic[i].append(DaviesBouldin(dm, kmedoids.labels_))

#             pbar.update(1)
#         kmedoids = KMedoids(n_clusters=n_cluster, random_state=int(seed), metric='precomputed', init='k-medoids++').fit(tes_gt[str(seed)].item()['dm'])
#         #gt_inertia_test.append(kmedoids.inertia_)
#         #gt_inertia_test.append(silhouette(tes_gt[str(seed)].item()['dm'], kmedoids.labels_))
#         gt_inertia_test.append(DaviesBouldin(tes_gt[str(seed)].item()['dm'], kmedoids.labels_))

#     val_mean_list = []
#     val_std_list = []

#     for i in range(len(pred_path)):
#         m_s = mean_std(model_inertia_dic[i])
#         val_mean_list.append(m_s[0])
#         val_std_list.append(m_s[1])

#     if gt_inertia_test:
#         m_s = mean_std(gt_inertia_test)

#     #print(val_mean_list)
#     #print(val_std_list)
#     #print(m_s)
    
#     plt.subplot(3, 3, idx // 3 * 3 + idx % 3 + 1)
#     plt.plot(train_nviews, val_mean_list, "-o")
#     plt.fill_between(train_nviews, [val_mean_list[i] - val_std_list[i] for i in range(len(val_mean_list))], [val_mean_list[i] + val_std_list[i] for i in range(len(val_mean_list))], facecolor='gray', alpha=0.2)
#     plt.plot(train_nviews, len(train_nviews) * [m_s[0]])
#     plt.fill_between(train_nviews, len(train_nviews) * [m_s[0]-m_s[1]],  len(train_nviews) * [m_s[0]+m_s[1]], facecolor='gray', alpha=0.2)
#     plt.xticks(train_nviews)
#     plt.legend([f"Pred {criterion} K={n_cluster}", f"GT {criterion} K={n_cluster}"])
#     if idx // 3 == 0:
#         plt.title(f"Pred PointCloud {criterion} Value")
#     if idx // 3 == 2:
#         plt.xlabel("Num of views per shape in Train Set")
#     if idx % 3 == 0:
#         plt.ylabel(f"{criterion} value")

# plt.savefig(f'scripts/shapenet13_pred/{criterion}_{n_cluster_list[0]}_{n_cluster_list[-1]}.png')




## HSIC 
# model_hsic_dic = {}
# for i in range(len(pred_path)):
#     model_hsic_dic.update({i: {'value':[], 'th': []} })

# gt_hsic_test = {'value':[], 'th': []}

# pbar = tqdm.tqdm(total=len(pred_path) * 5)

# for seed in ['1', '2', '3', '4', '5']:
#     for i in range(len(pred_path)):
#         pred_gt = np.load(pred_path[i], allow_pickle=True)
#         dm = pred_gt[seed].item()['dm']
#         value, threshold = hsic_gam_mat(dm, dm, 0.1)
#         model_hsic_dic[i]['value'].append(value)
#         model_hsic_dic[i]['th'].append(threshold)
#         pbar.update(1)
#     dm = tes_gt[seed].item()['dm']
#     value, threshold = hsic_gam_mat(dm, dm, 0.5)
#     gt_hsic_test['value'].append(value)
#     gt_hsic_test['th'].append(threshold)

# val_mean_list = []
# val_std_list = []
# th_mean_list = []
# th_std_list = []

# for i in range(len(pred_path)):
#     print(pred_path[i])
#     val_m_s = mean_std(model_hsic_dic[i]['value'])
#     th_m_s = mean_std(model_hsic_dic[i]['th'])
#     val_mean_list.append(val_m_s[0])
#     val_std_list.append(val_m_s[1])
#     th_mean_list.append(th_m_s[0])
#     th_std_list.append(th_m_s[1])

# val_mean_list = [round(item, 6) for item in val_mean_list]
# val_std_list = [round(item, 6) for item in val_std_list]
# th_mean_list = [round(item, 6) for item in th_mean_list]
# th_std_list = [round(item, 6) for item in th_std_list]

# print(val_mean_list)
# print(val_std_list)
# print(th_mean_list)
# print(th_std_list)

# print(mean_std(gt_hsic_test['value']))
# print(mean_std(gt_hsic_test['th']))



####################################################################################################
###
inertia_mean=args.inertia_mean

plt.figure(figsize=(12, 6))
dataset = 'table_car'  #shapenet13   table_car
file_path = f'scripts/{dataset}/oc_vc_sscore/dm_part_sscore'
#file_path = f'scripts/{dataset}/pts_testgt_dm_part_sscore'
plt_path = os.path.join(file_path, 'inertia_plots')
Path(plt_path).mkdir(parents=True, exist_ok=True)
obj_score_list = []
view_score_list = []
sample_num_list = [20, 50, 100, 200, 500, 1000, 2000]  #, 100, 200, 500, 
pbar = tqdm.tqdm(total=len(sample_num_list) * 5)
for sample_num in sample_num_list:
    args.sample_num = sample_num
    checkpoints = glob.glob(os.path.join(file_path, f'*_{args.sample_num}.npz'))
    checkpoints.sort()
    obj_stats = np.load(checkpoints[0], allow_pickle=True)
    view_stats = np.load(checkpoints[1], allow_pickle=True)
    obj_inertia_list = []
    view_inertia_list = []
    for seed in ['1', '2', '3', '4', '5']:
        obj_dm = obj_stats[seed].item()['dm']
        view_dm = view_stats[seed].item()['dm']
        obj_inertia, matrix_part, part_preference = inertia_ap(obj_dm, seed=int(seed), pc=args.perf_perc, normalize=args.normalize)
        view_inertia, matrix_part, part_preference = inertia_ap(view_dm, seed=int(seed), pc=args.perf_perc, normalize=args.normalize)

        if inertia_mean:
            obj_inertia_list.append(obj_inertia / obj_dm.shape[0])
            view_inertia_list.append(view_inertia / view_dm.shape[0])
        else:
            obj_inertia_list.append(obj_inertia)
            view_inertia_list.append(view_inertia)
        pbar.update(1)

    obj_score_list.append(mean_std(obj_inertia_list))
    view_score_list.append(mean_std(view_inertia_list))


plt.plot(sample_num_list, [item[0] for item in obj_score_list], 'ro-', label='Object Center')
plt.fill_between(sample_num_list, [item[0]-item[1] for item in obj_score_list], [item[0]+item[1] for item in obj_score_list], facecolor='red', alpha=0.2)
plt.plot(sample_num_list, [item[0] for item in view_score_list], 'bo-', label='Viewer Center')
plt.fill_between(sample_num_list, [item[0]-item[1] for item in view_score_list], [item[0]+item[1] for item in view_score_list], facecolor='blue', alpha=0.2)
plt.legend()
plt.xlabel("Number of Samples", fontsize=15)
plt.xticks(sample_num_list)
plt.ylabel("Inertia", fontsize=15)

if inertia_mean:
    plt.suptitle(f"{dataset} 13 OC/VC GT Shape Inertia Mean AP perf {args.perf_perc}")
    plt.savefig(os.path.join(plt_path, f'inertia_mean_ap_perf{args.perf_perc}_norm{args.normalize}.png'))
else:
    plt.suptitle(f"{dataset} 13 OC/VC GT Shape Inertia Sum AP perf {args.perf_perc}")
    plt.savefig(os.path.join(plt_path, f'inertia_sum_ap_perf{args.perf_perc}_norm{args.normalize}.png'))

#plt.show()



