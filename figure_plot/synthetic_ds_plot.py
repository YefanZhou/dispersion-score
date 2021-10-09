import os
import sys
import torch
from os.path import join
from pathlib import Path
import glob
import matplotlib.pyplot as plt 
import numpy as np
sys.path.append("../../")
from eval.eval_utils import mean_std

TRAIN_SAMPLE_SIZE = 20
TEST_SAMPLE_SIZE = 200
BASE_DIR = 'eval/eval_results/eval_ds_synthetic'
save_dir = "figure_plot/figures/synthetic/metric_curve"
Path(save_dir).mkdir(parents=True, exist_ok=True)

train_stats_path = f"{BASE_DIR}/checkpoints_1"
train_stats_dirs = glob.glob(join(train_stats_path, '*_numsample20.npz'))
train_stats_dirs.sort()

aug_mode_lst = ["image_aug", "shape_aug"]   # "shape_aug"

train_path_dic = {"image_aug":{"image":[], "points":[]}, "shape_aug":{"image":[], "points":[]}}
train_data_dic = {"image_aug":{"image":{}, "points":{}}, "shape_aug":{"image":{}, "points":{}}}
data_type_list = ["image", "points"]

cltsize_list = [1, 5, 10, 15, 20, 25, 30, 35]
data_indexes = list(range(1, 9))
fontsize = 60
linewidth = 7
markersize = 30

key_list = ["KMedoids_Inertia_k2p0"]  #"chamfer_stats", "fscore_stats"

for key in key_list:
    for aug_mode in aug_mode_lst:
        for data_type in data_type_list:
            train_data_dic[aug_mode][data_type].update({key:[]})

counter = 0
for data_type in data_type_list:
    for aug_mode in aug_mode_lst:
        tmp_aug_lst = []
        for train_stats_dir in train_stats_dirs:
            if aug_mode in train_stats_dir:
                tmp_aug_lst.append(train_stats_dir)
                counter += 1
        for cltsize in cltsize_list:
            substring = f"_train_{data_type}_cluster_{aug_mode}_2by10_cltsize{cltsize:02}"
            train_path_dic[aug_mode][data_type].append(next((s for s in tmp_aug_lst if substring in s), None))

print(train_path_dic)
plt.style.use('ggplot')
for data_type in data_type_list:
    for aug_mode in aug_mode_lst:
        for key in key_list:
            for clt_path in train_path_dic[aug_mode][data_type]:
                stats = np.load(clt_path, allow_pickle=True)
                train_data_dic[aug_mode][data_type][key].append([stats[key][0]/stats["sample_num"], stats[key][1]/stats["sample_num"]])


pred_stats_path = f"{BASE_DIR}/checkpoints_pred"
pred_stats_dirs = glob.glob(join(pred_stats_path, '*_pred_*'))
pred_stats_dirs.sort()



pred_path_dic = {"image_aug":[], "shape_aug":[]}
pred_data_dic = {"image_aug":{}, "shape_aug":{}}

cltsize_list = [1, 5, 10, 15, 20, 25, 30, 35]
data_indexes = list(range(1, 9))
data_indexes_lessimg_ticks = [r'$S_{1}I_{8}$',r'$S_{1}I_{7}$',r'$S_{1}I_{6}$',r'$S_{1}I_{5}$', r'$S_{1}I_{4}$', 
                            r'$S_{1}I_{3}$',r'$S_{1}I_{2}$',r'$S_{1}I_{1}$']
data_indexes_moreshape_ticks = [r'$S_{1}I_{8}$',r'$S_{2}I_{8}$',r'$S_{3}I_{8}$',r'$S_{4}I_{8}$', r'$S_{5}I_{8}$', 
                            r'$S_{6}I_{8}$',r'$S_{7}I_{8}$',r'$S_{8}I_{8}$']                    
seed_list = [1]



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

print(pred_path_dic)

plt.style.use('ggplot')
for aug_mode in aug_mode_lst:
    for key in key_list:
        for clt_tuple in pred_path_dic[aug_mode]:
            tmp_list = []
            for seed_data in clt_tuple:
                stats = np.load(seed_data, allow_pickle=True)
                tmp_list.append(stats[key][0]/stats["sample_num"])
            pred_data_dic[aug_mode][key].append(mean_std(tmp_list))

test_gt_stats_path = join("eval/eval_results/eval_ds_synthetic/checkpoints_1", 
        "stats_object_test_points_cluster_image_aug_2by10_cltsize01_numsample200.npz")

test_gt_stats = np.load(test_gt_stats_path)

test_gt_inertia = test_gt_stats[key_list[0]][0]/test_gt_stats["sample_num"]

data_indexes_lessimg_ticks.reverse()
for aug_mode in aug_mode_lst: 
    plt.figure(figsize=(16, 35))
    for idx, data_type in enumerate(data_type_list):
        for key in key_list:
            #print(aug_mode, data_type, key)
            plt.subplot(3, 1, idx+1)
            if idx == 0:
                if aug_mode == "image_aug":
                    plt.suptitle("Less Image Augmentation", fontsize=fontsize+10, x=0.53, y=1.01)
                elif aug_mode == "shape_aug":
                    plt.suptitle("More Shape Augmentation", fontsize=fontsize+10, x=0.53, y=1.01)
            if data_type == 'image':
                color = 'goldenrod'
                plt.ylim([-0.1, 22])
                plt.yticks(np.arange(0, 22, 5))
            elif data_type == 'points':
                color = 'mediumblue'
                plt.ylim([1.6e-3, 7e-3])
            plt.plot(data_indexes, [item[0] for item in train_data_dic[aug_mode][data_type][key]], 'o-', c=color, linewidth=linewidth, markersize=markersize)
            plt.fill_between(data_indexes, [item[0]-item[1] for item in train_data_dic[aug_mode][data_type][key]], 
                [item[0]+item[1] for item in train_data_dic[aug_mode][data_type][key]], 
                facecolor='b', alpha=0.2)

            plt.yticks(fontsize=fontsize-10)
            
            if data_type == 'image':
                plt.legend(["Training Image"],fontsize=fontsize-10, loc="lower left")
                if aug_mode == 'image_aug':
                    plt.ylabel('Input DS (Image)',fontsize=fontsize, labelpad=20)
                else:
                    plt.ylabel('    ',fontsize=fontsize, labelpad=20)
            elif data_type == 'points':
                plt.legend(["Training Shape"],fontsize=fontsize-10, loc="upper left")
                if aug_mode == 'image_aug':
                    plt.ylabel('Input DS (Shape)',fontsize=fontsize, labelpad=40)
                else:
                    plt.ylabel('   ',fontsize=fontsize, labelpad=40)

            if aug_mode == 'image_aug':
                plt.xticks(data_indexes, data_indexes_lessimg_ticks, fontsize=fontsize-5)
                plt.xlim(max(data_indexes)+0.5, min(data_indexes)-0.5)
            elif aug_mode == 'shape_aug':
                plt.xticks(data_indexes, data_indexes_moreshape_ticks, fontsize=fontsize-5)
                plt.xlim(min(data_indexes)-0.5, max(data_indexes)+0.5)

            ax = plt.gca()
            #'sci'
            plt.ticklabel_format(axis="y", style="sci", scilimits=(-1,2))
            ax.yaxis.offsetText.set_fontsize(fontsize-10)
            ax.xaxis.offsetText.set_fontsize(fontsize-10)

    plt.subplot(3, 1, idx+2)
    plt.plot(data_indexes, [item[0] for item in pred_data_dic[aug_mode][key]], 'o-', c='darkcyan', linewidth=linewidth, markersize=markersize) 
    plt.plot(data_indexes, len(data_indexes) * [test_gt_inertia], '--', c='r', linewidth=linewidth, markersize=markersize)
    plt.fill_between(data_indexes, [item[0]-item[1] for item in pred_data_dic[aug_mode][key]], 
        [item[0]+item[1] for item in pred_data_dic[aug_mode][key]], 
        facecolor='b', alpha=0.2)
    plt.xticks(data_indexes)
    if aug_mode == 'image_aug':
        plt.xticks(data_indexes, data_indexes_lessimg_ticks, fontsize=fontsize-5)
        plt.xlim(max(data_indexes)+0.5, min(data_indexes)-0.5)
        
    elif aug_mode == 'shape_aug':
        plt.xticks(data_indexes, data_indexes_moreshape_ticks, fontsize=fontsize-5)
        plt.xlim(min(data_indexes)-0.5, max(data_indexes)+0.5)
        
    plt.legend(["Reconstruction", "GT"],fontsize=fontsize-10, loc='upper left')
    plt.ylim([1e-3, 7.5e-3])
        
    plt.yticks(fontsize=fontsize-10)
    ax = plt.gca()
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.yaxis.get_major_formatter().set_powerlimits((0,1))
    ax.yaxis.offsetText.set_fontsize(fontsize-10)
    ax.xaxis.offsetText.set_fontsize(fontsize-10)
    if aug_mode == 'image_aug':
        plt.ylabel('Output DS',fontsize=fontsize, labelpad=40)
    else:
        plt.ylabel('  ',fontsize=fontsize, labelpad=40)

    
    plt.tight_layout()
    plt.subplots_adjust(hspace = 0.2)
    plt.savefig(join(save_dir, f"gt_train_pred_augmode{aug_mode}_{key}.png"), bbox_inches='tight')

    


        
    







        





