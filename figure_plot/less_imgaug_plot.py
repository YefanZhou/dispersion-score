import os
import sys
import glob
import numpy as np
from os.path import join
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
recons_color = "darkcyan"
gt_color = 'red'
fontsize = 60
linewidth = 7
markersize = 30
parser = argparse.ArgumentParser()

parser.add_argument("--split", type=str, default='pred', help='')
opt = parser.parse_args()
save_dir = "figure_plot/figures/shapenet/less_imgaug"
Path(save_dir).mkdir(parents=True, exist_ok=True)
xticks = [0, 15, 30, 45, 60, 75, 90]
key_to_scale = {'KMedoids_Inertia_k500p0':1000,  'fscore_stats':1, 'chamfer_stats':1000}
samplenum_key = 'sample_num'
pred_base_dir = 'eval/eval_results/lessimg_aug'
if opt.split == 'pred':
    key_lst = ['KMedoids_Inertia_k500p0', 'fscore_stats', 'chamfer_stats']

    
    pred_npz_lst = []
    for anglerange in xticks:
        pred_npz_lst.append(glob.glob(join(pred_base_dir, f"*yawrange{anglerange:03}*")))

    stats_dic = {}
    for key in key_lst:
        stats_dic.update({key:[]})
        for _ in range(len(pred_npz_lst)):
            stats_dic[key].append([])

    for idx, seed_tuple in enumerate(pred_npz_lst):
        for npz_file in seed_tuple:
            stats_data = np.load(npz_file)
            for key in key_lst:
                assert stats_data[samplenum_key] == 8762
                if "Inertia" in key:
                    stats_dic[key][idx].append(stats_data[key][0]/stats_data[samplenum_key])
                else:
                    stats_dic[key][idx].append(stats_data[key][0])
    
    print(stats_dic)
    test_gt_points_path = glob.glob(join(pred_base_dir, "*test_points*"))[0]
    test_gt_points_stats = np.load(test_gt_points_path)

    test_gt_points_inertia_mean = test_gt_points_stats[key_lst[0]][0] / test_gt_points_stats['sample_num']
    plt.style.use('ggplot')
    for idx, key in enumerate(key_lst):
        _, ax1 = plt.subplots(figsize=(18,14))
        print(stats_dic[key])
        lns1 = ax1.plot(xticks, [item[0] for item in stats_dic[key]], 'o-', c=recons_color, label="Reconstruction", linewidth=linewidth, markersize=markersize)
        ax1.fill_between(xticks, [item[0]-item[1] for item in stats_dic[key]], 
        [item[0]+item[1] for item in stats_dic[key]], 
        facecolor=recons_color, alpha=0.3) 
        for x, y in zip(xticks, [item[0] for item in stats_dic[key]]):
            ax1.text(x, y-0.0002, f"{y*key_to_scale[key]:.2f}", color="black", fontsize=fontsize-20)
        
        if "Inertia" in key:
            lns2 = ax1.plot(xticks, len(xticks) * [test_gt_points_inertia_mean], '--', c=gt_color, label="GT", linewidth=linewidth)
            lns = lns1+lns2
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, fontsize=fontsize, loc='center left')
        else:
            if "chamfer" in key:
                lns = lns1
                labs = [l.get_label() for l in lns]
                ax1.legend(lns, labs, fontsize=fontsize, loc='upper left')
            else:
                lns = lns1
                labs = [l.get_label() for l in lns]
                ax1.legend(lns, labs, fontsize=fontsize, loc='lower left') 

        ax1.set_xlabel(r'Angle range of rendering viewpoint $\alpha$', fontsize=fontsize, labelpad=20)
        if key == 'KMedoids_Inertia_k500p0':
            ax1.set_ylabel('Output DS', fontsize=fontsize)
            ax1.set_ylim([2.5e-3, 6.0e-3])
            ax1.set_yticks(np.arange(2.5e-3, 6.5e-3, 1e-3))
        elif key == 'fscore_stats':
            ax1.set_ylabel('F-score', fontsize=fontsize)
        elif key == 'chamfer_stats':
            ax1.set_ylabel('CD', fontsize=fontsize)

        #ax1.set_ylim([2.1e-3, 6.0e-3])
        #ax1.set_yticks(np.arange(2.5e-3, 6.5e-3, 1e-3))
        ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        #ax1.yaxis.get_major_formatter().set_powerlimits((-2,2))
        ax1.tick_params(axis="x", labelsize=fontsize-10)
        ax1.tick_params(axis="y", labelsize=fontsize-10)
        ax1.set_xlim(max(xticks)+10, min(xticks)-10) 
        ax1.set_xticks(xticks) 
        ax1.yaxis.offsetText.set_fontsize(fontsize-10)
        ax1.xaxis.offsetText.set_fontsize(fontsize-10)
        #ax1.set_ylim([2.0e-3, 7.0e-3])

        plt.tight_layout()
        save_path = join(save_dir, f'pred_{key}.png')
        plt.savefig(save_path, bbox_inches='tight')

elif opt.split == 'train':
    imagecolor = 'goldenrod'
    shapecolor = 'mediumblue'

    key_lst = ['KMedoids_Inertia_k500p0']
    train_img_npz_lst = glob.glob(join(pred_base_dir, "*train_image*"))
    stats_dic = {}
    for key in key_lst:
        stats_dic.update({key:[]})
        for _ in range(len(train_img_npz_lst)):
            stats_dic[key].append([])
    
    for idx, train_img_npz in enumerate(train_img_npz_lst):
        stats_data = np.load(train_img_npz)
        for key in key_lst:
            assert stats_data[samplenum_key] == 35021
            if "Inertia" in key:
                stats_dic[key][idx].append(stats_data[key][0]/stats_data[samplenum_key])
            else:
                stats_dic[key][idx].append(stats_data[key][0])

    oc_trainpoints_stats = np.load("../scripts/shapenet13_benchmark/checkpoints_gt/object_train_points_trainnv01_testnv01_35021.npz")

    oc_shape_inertia_meanstd = oc_trainpoints_stats[key_lst[0]][0]
    assert oc_trainpoints_stats[samplenum_key] == 35021
    oc_shape_inertia = oc_shape_inertia_meanstd/oc_trainpoints_stats[samplenum_key]

    stats_dic = np.load(join('stats_cache', 'oc_angle_limit_trainimg.npy'), allow_pickle=True)
    stats_dic = stats_dic.item()
    oc_trainpoints_stats = np.load("../scripts/shapenet13_benchmark/checkpoints_gt/object_train_points_trainnv01_testnv01_35021.npz")

    oc_shape_inertia_meanstd = oc_trainpoints_stats[key_lst[0]][0]
    assert oc_trainpoints_stats[samplenum_key] == 35021
    oc_shape_inertia = oc_shape_inertia_meanstd/oc_trainpoints_stats[samplenum_key]
    print(stats_dic)
    print(oc_shape_inertia, oc_trainpoints_stats[samplenum_key])

    plt.style.use('ggplot')
    for idx, key in enumerate(key_lst):
        _, ax1 = plt.subplots(figsize=(18,14))
        ax2 = ax1.twinx()
        ax1.set_xlabel(r'Angle range of rendering viewpoint $\alpha$', fontsize=fontsize, labelpad=20)
        ax1.set_ylabel('Input DS (Image)', fontsize=fontsize)
        data_lst = stats_dic[key] 
        lns1 = ax1.plot(xticks, [item[0] for item in data_lst], 'o-', c=imagecolor, label="Training Image", linewidth=linewidth, markersize=markersize)
        lns2 = ax2.plot(xticks, len(xticks) * [oc_shape_inertia], 'o-', c=shapecolor,label="Training Shape", linewidth=linewidth, markersize=markersize)
        ax1.set_xticks(xticks)
        ax1.tick_params(axis="x", labelsize=fontsize-10)
        ax1.tick_params(axis="y", labelsize=fontsize-10, colors=imagecolor)
        ax1.set_xlim(max(xticks)+10, min(xticks)-10) 
        
        ax2.set_ylim([2.5e-3, 6.0e-3])
        ax2.set_yticks(np.arange(2.5e-3, 6.5e-3, 1e-3))
        ax2.tick_params(axis="y", labelsize=fontsize-10, colors=shapecolor)
        ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        ax2.yaxis.get_major_formatter().set_powerlimits((0,1))
        ax2.set_ylabel('Input DS (Shape)', fontsize=fontsize)
        ax2.yaxis.offsetText.set_fontsize(fontsize-10)
        ax2.xaxis.offsetText.set_fontsize(fontsize-10)
        
        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns,labs, fontsize=fontsize, loc="lower left")

        
        plt.tight_layout()
        save_path = f'cache/oc_angle_limit/train_{key}.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

# train_pts_dir = \
# ['anglelimit_yawrangeShapeNetV1RenderingSampleYaw000_viewer_train_points_rgb_RGB_mag0_magnitude0_n_op0_prob0_trainnv01_testnv01_35021.npz', 
# 'anglelimit_yawrangeShapeNetV1RenderingSampleYaw015_viewer_train_points_rgb_RGB_mag0_magnitude0_n_op0_prob0_trainnv01_testnv01_35021.npz',
# 'anglelimit_yawrangeShapeNetV1RenderingSampleYaw030_viewer_train_points_rgb_RGB_mag0_magnitude0_n_op0_prob0_trainnv01_testnv01_35021.npz',
# 'anglelimit_yawrangeShapeNetV1RenderingSampleYaw045_viewer_train_points_rgb_RGB_mag0_magnitude0_n_op0_prob0_trainnv01_testnv01_35021.npz',
# 'anglelimit_yawrangeShapeNetV1RenderingSampleYaw060_viewer_train_points_rgb_RGB_mag0_magnitude0_n_op0_prob0_trainnv01_testnv01_35021.npz',
# 'anglelimit_yawrangeShapeNetV1RenderingSampleYaw075_viewer_train_points_rgb_RGB_mag0_magnitude0_n_op0_prob0_trainnv01_testnv01_35021.npz',
# 'anglelimit_yawrangeShapeNetV1RenderingSampleYaw090_viewer_train_image_rgb_RGB_mag0_magnitude0_n_op0_prob0_trainnv01_testnv01_35021.npz']

