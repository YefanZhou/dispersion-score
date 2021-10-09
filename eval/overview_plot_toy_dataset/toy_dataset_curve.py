import os
import sys
import torch
from os.path import join
import glob
import matplotlib.pyplot as plt 
import numpy as np
sys.path.append("../../")
import matplotlib
from eval.eval_utils import mean_std

TRAIN_NUM_SAMPLE = 20
TEST_NUM_SAMPLE = 200


pred_stats_path = "../scripts/shapenet13_toydata/checkpoints_pred_lastepoch"
pred_stats_dirs = glob.glob(join(pred_stats_path, '*_cluster_*'))
pred_stats_dirs.sort()

aug_mode_lst = ["shape_aug"] #"image_aug"
#aug_mode_lst = ["shape_aug"]
pred_path_dic = {"image_aug":[], "shape_aug":[]}
pred_data_dic = {"image_aug":{}, "shape_aug":{}}

cltsize_list = [1, 5, 10, 15, 20, 25, 30, 35]
data_indexes = list(range(1, 9))
seed_list = [1]

#key_list = ["KMedoids_Inertia_k2p0", "chamfer_stats", "fscore_stats"]
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


base_save_path = "results/Metric_Curve/Prediction"
#plt.style.use('ggplot')
for aug_mode in aug_mode_lst:
    for key in key_list:
        for clt_tuple in pred_path_dic[aug_mode]:
            tmp_list = []
            for seed_data in clt_tuple:
                stats = np.load(seed_data, allow_pickle=True)
                ## GET Average 
                tmp_list.append(stats[key][0]/TEST_NUM_SAMPLE)
            pred_data_dic[aug_mode][key].append(mean_std(tmp_list))
            
    #for key in key_list:
        #plt.plot(data_indexes, [item[0] for item in pred_data_dic[aug_mode][key]], 'o-', c='b')
        #plt.fill_between(data_indexes, [item[0]-item[1] for item in pred_data_dic[aug_mode][key]], 
        #    [item[0]+item[1] for item in pred_data_dic[aug_mode][key]], 
        #    facecolor='b', alpha=0.2)
        #plt.xticks(data_indexes)
        
        # if key == "chamfer_stats":
        #     plt.ylabel("Chamfer", fontsize=18, fontweight='bold')
        # elif key == "fscore_stats":
        #     plt.ylabel("F-score", fontsize=18, fontweight='bold')
        # elif key == "KMedoids_Inertia_k2p0":
        #     plt.ylabel("Inertia", fontsize=18, fontweight='bold')
        # if aug_mode == 'image_aug':
        #     final_save_path = join(base_save_path, "ImageAug")
        #     plt.xlabel(r'$j$', fontsize=18, fontweight='bold')
        #     plt.xlim(max(data_indexes)+0.5, min(data_indexes)-0.5)
        # elif aug_mode == 'shape_aug':
        #     final_save_path = join(base_save_path, "ShapeAug")
        #     plt.xlabel(r'$i$', fontsize=18, fontweight='bold')
        #     plt.xlim(min(data_indexes)-0.5, max(data_indexes)+0.5)
        
        # plt.title("Recons", fontsize=20, fontweight='bold')

        # plt.tight_layout()
        # #plt.savefig(join(final_save_path, f"pred_{aug_mode}_{key}.png"), bbox_inches='tight')
        # #plt.close()
        # plt.show()



train_stats_path = "../scripts/shapenet13_toydata/checkpoints_1"
train_stats_dirs = glob.glob(join(train_stats_path, '*_numsample20.npz'))
train_stats_dirs.sort()


aug_mode_lst = ["shape_aug"]  #"image_aug"
#aug_mode_lst = ["shape_aug"]
train_path_dic = {"image_aug":{"image":[], "points":[]}, "shape_aug":{"image":[], "points":[]}}
train_data_dic = {"image_aug":{"image":{}, "points":{}}, "shape_aug":{"image":{}, "points":{}}}
data_type_list = ["points"]  #"image", 

cltsize_list = [1, 5, 10, 15, 20, 25, 30, 35]
data_indexes = list(range(1, 9))


key_list = ["KMedoids_Inertia_k2p0"]
#key_list = ["chamfer_stats", "fscore_stats"]

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

save_path_dic = {"image":"results/Metric_Curve/TrainImage", "points":"results/Metric_Curve/TrainPoint"}

#plt.style.use('ggplot')
for data_type in data_type_list:
    for aug_mode in aug_mode_lst:
        for key in key_list:
            for clt_path in train_path_dic[aug_mode][data_type]:
                stats = np.load(clt_path, allow_pickle=True)
                ## GET Average 
                train_data_dic[aug_mode][data_type][key].append(stats[key][0]/TRAIN_NUM_SAMPLE)


test_gt_stats_path = join("../scripts/shapenet13_toydata/checkpoints_1", 
        "stats_viewer_test_points_cluster_shape_aug_2by10_cltsize01_numsample200.npz")

test_gt_stats = np.load(test_gt_stats_path)
test_gt_inertia = test_gt_stats[key_list[0]][0]/TEST_NUM_SAMPLE

for aug_mode in aug_mode_lst: 
    for data_type in data_type_list:
        for key in key_list:
            plt.figure(figsize=(35, 30))
            print(aug_mode, data_type, key)
            input_inertia = [item for item in train_data_dic[aug_mode][data_type][key]]
            out_inertia = [item[0] for item in pred_data_dic[aug_mode][key]]
            plt.plot(input_inertia, out_inertia, 'o-', c='darkcyan', markersize=40, linewidth=7)
            plt.plot(input_inertia, len(input_inertia) * [test_gt_inertia], '--', c='orange', linewidth=14)
            #plt.fill_between(data_indexes, [item[0]-item[1] for item in train_data_dic[aug_mode][data_type][key]], 
            #    [item[0]+item[1] for item in train_data_dic[aug_mode][data_type][key]], 
            #    facecolor='b', alpha=0.2)
            
            plt.ylabel("Output Dispersion Score", fontsize=100, labelpad=50)
            if aug_mode == 'image_aug':
                final_save_path = "plot"
                plt.xlabel("Input Dispersion Score", fontsize=100, labelpad=50)
                
            elif aug_mode == 'shape_aug':
                final_save_path = "plot"
                plt.xlabel("Input Dispersion Score", fontsize=100, labelpad=50)

            ax = plt.gca()
            #'sci'
            ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='both')
            ax.yaxis.offsetText.set_fontsize(45)
            ax.xaxis.offsetText.set_fontsize(45)
            plt.legend(["Neural Network", "Ground Truth"], loc='upper center', bbox_to_anchor=(0.5, 1.10), fancybox=True, shadow=True, fontsize=70, ncol=2)
            plt.xticks(fontsize=60)
            plt.yticks(fontsize=60)

            plt.tight_layout()
            plt.savefig(join(final_save_path, f"train_datatype{data_type}_augmode{aug_mode}_versus_prediction_{key}.png"), bbox_inches='tight')
            #plt.show()
            #plt.close()
    







        





