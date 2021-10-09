import os
import sys
import torch
from os.path import join
import glob
import matplotlib.pyplot as plt 
import numpy as np
sys.path.append("../../")
import matplotlib
from sklearn.manifold import TSNE
import argparse
from eval.eval_utils import mean_std
from auxiliary.my_utils import visuaize_pts
from auxiliary.PointFlowRenderer.pointflow_fig_function import pts_to_xml

TRAIN_NUM_SAMPLE = 20
TEST_NUM_SAMPLE = 200
# clsize 01  20,40, 150, 180
# clsize 30  60,90,100,120 
selected_index_list = [60, 90, 100, 120] #,  [20, 40, 150, 180]
color_list = ['orange', 'red', 'green', 'brown', 'magenta', 'purple', 'darkcyan', 'cyan']

parser = argparse.ArgumentParser(
    description='sum the integers at the command line')
parser.add_argument('--seed', type=int, default=1, help='')
args = parser.parse_args()

tsne = TSNE(n_components=2, metric='precomputed', random_state=args.seed)

pred_stats_path = "../scripts/shapenet13_toydata/checkpoints_pred_lastepoch"
log_prediction_path = "../../log/toy_data"
prediction_fname = "prediction.npy"
pointcloud_vis_folder = os.path.abspath("./plot/pts_vis")
pred_stats_dirs = glob.glob(join(pred_stats_path, '*_cluster_*'))
pred_stats_dirs.sort()

aug_mode_lst = ["shape_aug"] #"image_aug" shape_aug
#aug_mode_lst = ["shape_aug"]
pred_path_dic = {"image_aug":[], "shape_aug":[]}
pred_data_dic = {"image_aug":{}, "shape_aug":{}}

cltsize_list = [1, 5, 10, 15, 20, 25, 30, 35]  #1, 5, 10, 15, 20, 25, 30 35
data_indexes = list(range(1, 9))
seed_list = [1]

key_list = ["KMedoids_Inertia_k2p0"]
#key_list = ["chamfer_stats", "fscore_stats"]

for key in key_list:
    for aug_mode in aug_mode_lst:
        pred_data_dic[aug_mode].update({key:[]})

counter = 0
for aug_mode in aug_mode_lst:
    tmp_aug_lst = []
    for pred_stats_dir in pred_stats_dirs:
        if aug_mode in pred_stats_dir:
            tmp_aug_lst.append(pred_stats_dir)
            counter += 1
    for cltsize in cltsize_list:
        pred_path_dic[aug_mode].append([])
        for seed in seed_list:
            substring = f"_cltsize{cltsize:02}_seed{seed}_"
            pred_path_dic[aug_mode][-1].append(next((s for s in tmp_aug_lst if substring in s), None))


base_save_path = "plot"
for aug_mode in aug_mode_lst:
    for key in key_list:
        for clt_tuple in pred_path_dic[aug_mode]:
            tmp_list = []
            for seed_data in clt_tuple:
                plot_savef = seed_data.split('/')[-1].replace('npz', 'png')
                
                ## visualize point cloud 

                total_string = seed_data.split('/')[-1].split('.')[0]
                log_folder = total_string[total_string.find('2021'):]
                predictions = np.load(join(log_prediction_path, log_folder, prediction_fname))
                for index in selected_index_list:
                    if index < 0:
                        index_string = f"negative{abs(index)}"
                    else:
                        index_string = f"{index}"
                    pts_to_xml(predictions[index], 
                    join(pointcloud_vis_folder, f"pred_{log_folder}_index{index_string}.xml"))
                    
                ## t-SNE visualization
                stats = np.load(seed_data, allow_pickle=True)
                dm_stats = stats['1'].item()
                dismat = dm_stats['dm']
                embeddings = tsne.fit_transform(dismat)

                plt.figure(figsize=(10,10))
                vis_x = embeddings[:, 0]
                vis_y = embeddings[:, 1]
                #selected_x = embeddings[selected_index_list, 0]
                #selected_y = embeddings[selected_index_list, 1]
                plt.scatter(vis_x, vis_y, s=800, c='b', alpha=0.6)
                #plt.scatter(selected_x, selected_y, s=800, c='r', alpha=1)
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(join(base_save_path, plot_savef))
                plt.close()

                #legend marker tsne
                # plt.figure(figsize=(5,5))
                # vis_x = embeddings[:, 0]
                # vis_y = embeddings[:, 1]
                # selected_x = embeddings[selected_index_list, 0]
                # selected_y = embeddings[selected_index_list, 1]
                # plt.scatter(vis_x, vis_y, s=40, c='b', alpha=0.3)
                # for idx, index in enumerate(selected_index_list):
                #     plt.scatter(embeddings[index, 0], embeddings[index, 1], s=40, c=color_list[idx], label=index)
                # plt.xticks([])
                # plt.yticks([])
                # plt.legend()
                # plt.tight_layout()
                # plt.savefig(join(base_save_path, "legend_"+plot_savef))
                # plt.close()


                ## GET Average 
                #tmp_list.append(stats[key][0]/TEST_NUM_SAMPLE)
            #pred_data_dic[aug_mode][key].append(mean_std(tmp_list))