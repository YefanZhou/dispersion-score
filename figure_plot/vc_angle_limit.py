import os
import sys
import numpy as np
from os.path import join
import argparse
sys.path.append("../../")
from eval.eval_utils import mean_std
import matplotlib.pyplot as plt

recons_color = "darkcyan"
gt_color = 'red'
fontsize = 60
linewidth = 7
markersize = 30
parser = argparse.ArgumentParser()
parser.add_argument("--preload", action='store_true', default=False)
parser.add_argument("--split", type=str, default='pred', help='')
opt = parser.parse_args()
xticks = [0, 15, 30, 45, 60, 75, 90]
key_to_scale = {'KMedoids_Inertia_k500p0':1000,  'fscore_stats':1, 'chamfer_stats':1000}
samplenum_key = 'sample_num'
if opt.split == 'pred':
    key_lst = ['KMedoids_Inertia_k500p0', 'fscore_stats', 'chamfer_stats']
    if not opt.preload:
        pred_base_dir = '../scripts/shapenet13_anglelimit/checkpoints_pred'
        pred_npz_lst = \
        [['anglelimit_yawrangeShapeNetV1RenderingSampleYaw090_viewer_pred_points_trainnv01_testnv01_8762_20210308_0118_anglelimit_yawrange000_viewer_atlasnetSPHERE1_seed1.npz',
        'anglelimit_yawrangeShapeNetV1RenderingSampleYaw090_viewer_pred_points_trainnv01_testnv01_8762_20210309_0209_anglelimit_yawrange000_viewer_atlasnetSPHERE1_seed2.npz',
        'anglelimit_yawrangeShapeNetV1RenderingSampleYaw090_viewer_pred_points_trainnv01_testnv01_8762_20210309_1048_anglelimit_yawrange000_viewer_atlasnetSPHERE1_seed3.npz'],
        ['anglelimit_yawrangeShapeNetV1RenderingSampleYaw090_viewer_pred_points_trainnv01_testnv01_8762_20210308_1642_anglelimit_yawrange015_viewer_atlasnetSPHERE1_seed1.npz',
        'anglelimit_yawrangeShapeNetV1RenderingSampleYaw090_viewer_pred_points_trainnv01_testnv01_8762_20210309_0607_anglelimit_yawrange015_viewer_atlasnetSPHERE1_seed2.npz',
        'anglelimit_yawrangeShapeNetV1RenderingSampleYaw090_viewer_pred_points_trainnv01_testnv01_8762_20210309_1049_anglelimit_yawrange015_viewer_atlasnetSPHERE1_seed3.npz'],
        ['anglelimit_yawrangeShapeNetV1RenderingSampleYaw090_viewer_pred_points_trainnv01_testnv01_8762_20210308_0554_anglelimit_yawrange030_viewer_atlasnetSPHERE1_seed1.npz',
        'anglelimit_yawrangeShapeNetV1RenderingSampleYaw090_viewer_pred_points_trainnv01_testnv01_8762_20210309_0613_anglelimit_yawrange030_viewer_atlasnetSPHERE1_seed2.npz',
        'anglelimit_yawrangeShapeNetV1RenderingSampleYaw090_viewer_pred_points_trainnv01_testnv01_8762_20210309_1111_anglelimit_yawrange030_viewer_atlasnetSPHERE1_seed3.npz'],
        ['anglelimit_yawrangeShapeNetV1RenderingSampleYaw090_viewer_pred_points_trainnv01_testnv01_8762_20210308_1643_anglelimit_yawrange045_viewer_atlasnetSPHERE1_seed1.npz',
        'anglelimit_yawrangeShapeNetV1RenderingSampleYaw090_viewer_pred_points_trainnv01_testnv01_8762_20210309_0615_anglelimit_yawrange045_viewer_atlasnetSPHERE1_seed2.npz',
        'anglelimit_yawrangeShapeNetV1RenderingSampleYaw090_viewer_pred_points_trainnv01_testnv01_8762_20210309_1507_anglelimit_yawrange045_viewer_atlasnetSPHERE1_seed3.npz'],
        ['anglelimit_yawrangeShapeNetV1RenderingSampleYaw090_viewer_pred_points_trainnv01_testnv01_8762_20210308_0940_anglelimit_yawrange060_viewer_atlasnetSPHERE1_seed1.npz',
        'anglelimit_yawrangeShapeNetV1RenderingSampleYaw090_viewer_pred_points_trainnv01_testnv01_8762_20210309_0638_anglelimit_yawrange060_viewer_atlasnetSPHERE1_seed2.npz',
        'anglelimit_yawrangeShapeNetV1RenderingSampleYaw090_viewer_pred_points_trainnv01_testnv01_8762_20210309_1519_anglelimit_yawrange060_viewer_atlasnetSPHERE1_seed3.npz'],
        ['anglelimit_yawrangeShapeNetV1RenderingSampleYaw090_viewer_pred_points_trainnv01_testnv01_8762_20210309_1941_anglelimit_yawrange075_viewer_atlasnetSPHERE1_seed1.npz',
        'anglelimit_yawrangeShapeNetV1RenderingSampleYaw090_viewer_pred_points_trainnv01_testnv01_8762_20210309_1955_anglelimit_yawrange075_viewer_atlasnetSPHERE1_seed2.npz',
        'anglelimit_yawrangeShapeNetV1RenderingSampleYaw090_viewer_pred_points_trainnv01_testnv01_8762_20210310_0335_anglelimit_yawrange075_viewer_atlasnetSPHERE1_seed3.npz'],
        ['anglelimit_yawrangeShapeNetV1RenderingSampleYaw090_viewer_pred_points_trainnv01_testnv01_8762_20210308_1644_anglelimit_yawrange090_viewer_atlasnetSPHERE1_seed1.npz',
        'anglelimit_yawrangeShapeNetV1RenderingSampleYaw090_viewer_pred_points_trainnv01_testnv01_8762_20210309_1037_anglelimit_yawrange090_viewer_atlasnetSPHERE1_seed2.npz',
        'anglelimit_yawrangeShapeNetV1RenderingSampleYaw090_viewer_pred_points_trainnv01_testnv01_8762_20210309_1521_anglelimit_yawrange090_viewer_atlasnetSPHERE1_seed3.npz']
        ]

        stats_dic = {}

        for key in key_lst:
            stats_dic.update({key:[]})
            for _ in range(len(pred_npz_lst)):
                stats_dic[key].append([])

        for idx, seed_tuple in enumerate(pred_npz_lst):
            for npz_file in seed_tuple:
                stats_data = np.load(join(pred_base_dir, npz_file))
                for key in key_lst:
                    assert stats_data[samplenum_key] == 8762
                    if "Inertia" in key:
                        stats_dic[key][idx].append(stats_data[key][0]/stats_data[samplenum_key])
                    else:
                        stats_dic[key][idx].append(stats_data[key][0])
        print(stats_dic)
        sys.exit(0)
        for key in key_lst:
            for idx, seed_tuple in enumerate(stats_dic[key]):
                stats_dic[key][idx] = mean_std(seed_tuple)
        print(stats_dic)
        np.save(join('stats_cache', 'vc_angle_limit_pred.npy'), stats_dic)

    else:
        #rendering angle
        stats_dic = np.load(join('stats_cache', 'vc_angle_limit_pred.npy'), allow_pickle=True)
        stats_dic = stats_dic.item()
        test_gt_points_path = "../scripts/shapenet13_benchmark/checkpoints_gt/viewer_test_points_rgb_RGB_mag0_n_op0_prob0_trainnv01_testnv01_8762.npz"
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
                ax1.text(x-2, y*1.00, f"{y*key_to_scale[key]:.2f}", color="black", fontsize=fontsize-20)
            
            if "Inertia" in key:
                #lns2 = ax1.plot(xticks, len(xticks) * [test_gt_points_inertia_mean], '--', c=gt_color, label="GT", linewidth=linewidth)
                #lns = lns1+lns2
                lns = lns1
                labs = [l.get_label() for l in lns]
                ax1.legend(lns, labs, fontsize=fontsize, loc='lower left')
            else:
                lns = lns1
                labs = [l.get_label() for l in lns]
                if key == 'fscore_stats':
                    ax1.legend(lns, labs, fontsize=fontsize, loc='lower left')
                else:
                    ax1.legend(lns, labs, fontsize=fontsize)

            ax1.set_xlabel(r'Angle range of rendering viewpoint $\alpha$', fontsize=fontsize, labelpad=20)
            if key == 'KMedoids_Inertia_k500p0':
                ax1.set_ylabel('Output DS', fontsize=fontsize)
                ax1.set_ylim([4.9e-3, 6.0e-3])
                ax1.set_yticks(np.arange(4.9e-3, 6.0e-3, 0.3e-3)) 
                ax1.yaxis.get_major_formatter().set_powerlimits((-3,-3))
            elif key == 'fscore_stats':
                ax1.set_ylabel('F-score', fontsize=fontsize)
            elif key == 'chamfer_stats':
                ax1.set_ylabel('CD', fontsize=fontsize)
                ax1.yaxis.get_major_formatter().set_powerlimits((-3,-3))

            #ax1.set_ylim([2.1e-3, 6.0e-3])
            #ax1.set_yticks(np.arange(2.5e-3, 6.5e-3, 1e-3))
            #ax1.ticklabel_format(axis="y", style="sci", scilimits=(-3,-3))
            #ax1.yaxis.get_major_formatter().set_powerlimits((-2,2))
            ax1.tick_params(axis="x", labelsize=fontsize-10)
            ax1.tick_params(axis="y", labelsize=fontsize-10)
            ax1.set_xlim(max(xticks)+10, min(xticks)-10) 
            ax1.set_xticks(xticks) 
            ax1.yaxis.offsetText.set_fontsize(fontsize-10)
            ax1.xaxis.offsetText.set_fontsize(fontsize-10)
            #ax1.set_ylim([2.0e-3, 7.0e-3])


            plt.tight_layout()
            save_path = f'cache/vc_angle_limit/pred_{key}.png'
            plt.savefig(save_path, bbox_inches='tight')

elif opt.split == 'train':
    imagecolor = 'goldenrod'
    shapecolor = 'mediumblue'
    train_base_dir = '../scripts/shapenet13_anglelimit/checkpoints_1'
    key_lst = ['KMedoids_Inertia_k500p0']
    train_img_npz_lst = \
    ['anglelimit_yawrangeShapeNetV1RenderingSampleYaw000_object_train_image_rgb_RGB_mag0_magnitude0_n_op0_prob0_trainnv01_testnv01_35021.npz', 
    'anglelimit_yawrangeShapeNetV1RenderingSampleYaw015_viewer_train_image_rgb_RGB_mag0_magnitude0_n_op0_prob0_trainnv01_testnv01_35021.npz',
    'anglelimit_yawrangeShapeNetV1RenderingSampleYaw030_viewer_train_image_rgb_RGB_mag0_magnitude0_n_op0_prob0_trainnv01_testnv01_35021.npz',
    'anglelimit_yawrangeShapeNetV1RenderingSampleYaw045_viewer_train_image_rgb_RGB_mag0_magnitude0_n_op0_prob0_trainnv01_testnv01_35021.npz',
    'anglelimit_yawrangeShapeNetV1RenderingSampleYaw060_viewer_train_image_rgb_RGB_mag0_magnitude0_n_op0_prob0_trainnv01_testnv01_35021.npz',
    'anglelimit_yawrangeShapeNetV1RenderingSampleYaw075_viewer_train_image_rgb_RGB_mag0_magnitude0_n_op0_prob0_trainnv01_testnv01_35021.npz',
    'anglelimit_yawrangeShapeNetV1RenderingSampleYaw090_viewer_train_image_rgb_RGB_mag0_magnitude0_n_op0_prob0_trainnv01_testnv01_35021.npz']

    train_pts_npz_lst = \
    ['anglelimit_yawrangeShapeNetV1RenderingSampleYaw000_viewer_train_points_rgb_RGB_mag0_magnitude0_n_op0_prob0_trainnv01_testnv01_35021.npz',
    'anglelimit_yawrangeShapeNetV1RenderingSampleYaw015_viewer_train_points_rgb_RGB_mag0_magnitude0_n_op0_prob0_trainnv01_testnv01_35021.npz',
    'anglelimit_yawrangeShapeNetV1RenderingSampleYaw030_viewer_train_points_rgb_RGB_mag0_magnitude0_n_op0_prob0_trainnv01_testnv01_35021.npz',
    'anglelimit_yawrangeShapeNetV1RenderingSampleYaw045_viewer_train_points_rgb_RGB_mag0_magnitude0_n_op0_prob0_trainnv01_testnv01_35021.npz',
    'anglelimit_yawrangeShapeNetV1RenderingSampleYaw060_viewer_train_points_rgb_RGB_mag0_magnitude0_n_op0_prob0_trainnv01_testnv01_35021.npz',
    'anglelimit_yawrangeShapeNetV1RenderingSampleYaw075_viewer_train_points_rgb_RGB_mag0_magnitude0_n_op0_prob0_trainnv01_testnv01_35021.npz',
    'anglelimit_yawrangeShapeNetV1RenderingSampleYaw090_viewer_train_points_rgb_RGB_mag0_magnitude0_n_op0_prob0_trainnv01_testnv01_35021.npz']
    if not opt.preload:
        stats_dic = {}
        for key in key_lst:
            stats_dic.update({key:[]})
            for _ in range(len(train_img_npz_lst)):
                stats_dic[key].append([])
        
        for idx, train_img_npz in enumerate(train_img_npz_lst):
            stats_data = np.load(join(train_base_dir, train_img_npz))
            for key in key_lst:
                assert stats_data[samplenum_key] == 35021
                if "Inertia" in key:
                    stats_dic[key][idx].append(stats_data[key][0]/stats_data[samplenum_key])
                else:
                    stats_dic[key][idx].append(stats_data[key][0])

        print(stats_dic)
        #np.save(join('stats_cache', 'vc_angle_limit_trainimg.npy'), stats_dic)

        stats_dic = {}
        for key in key_lst:
            stats_dic.update({key:[]})
            for _ in range(len(train_pts_npz_lst)):
                stats_dic[key].append([])
        
        for idx, train_pts_npz in enumerate(train_pts_npz_lst):
            stats_data = np.load(join(train_base_dir, train_pts_npz))
            for key in key_lst:
                assert stats_data[samplenum_key] == 35021
                if "Inertia" in key:
                    stats_dic[key][idx].append(stats_data[key][0]/stats_data[samplenum_key])
                else:
                    stats_dic[key][idx].append(stats_data[key][0])

        print(stats_dic)
        sys.exit(0)
        np.save(join('stats_cache', 'vc_angle_limit_trainpoints.npy'), stats_dic)
    
    else:
        stats_dic_img = np.load(join('stats_cache', 'vc_angle_limit_trainimg.npy'), allow_pickle=True)
        stats_dic_img = stats_dic_img.item()
        stats_dic_points = np.load(join('stats_cache', 'vc_angle_limit_trainpoints.npy'), allow_pickle=True)
        stats_dic_points = stats_dic_points.item()
        print(stats_dic_img, stats_dic_points)

        #oc_trainpoints_stats = np.load("../scripts/shapenet13_benchmark/checkpoints_gt/object_train_points_trainnv01_testnv01_35021.npz")
        #oc_shape_inertia_meanstd = oc_trainpoints_stats[key_lst[0]][0]
        #oc_shape_inertia = oc_shape_inertia_meanstd/oc_trainpoints_stats[samplenum_key]
        
        plt.style.use('ggplot')
        for idx, key in enumerate(key_lst):
            _, ax1 = plt.subplots(figsize=(18,14))
            ax2 = ax1.twinx()
            ax1.set_xlabel(r'Angle range of rendering viewpoint $\alpha$', fontsize=fontsize, labelpad=20)
            ax1.set_ylabel('Input DS (Image)', fontsize=fontsize)
            img_data_lst = stats_dic_img[key] 
            points_data_lst = stats_dic_points[key]
            lns1 = ax1.plot(xticks, [item[0] for item in img_data_lst], 'o-', c=imagecolor, label="Training Image", linewidth=linewidth, markersize=markersize)
            lns2 = ax2.plot(xticks, [item[0] for item in points_data_lst], 'o-', c=shapecolor,label="Training Shape", linewidth=linewidth, markersize=markersize)
            ax1.set_xticks(xticks)
            ax1.tick_params(axis="x", labelsize=fontsize-10)
            ax1.tick_params(axis="y", labelsize=fontsize-10, colors=imagecolor)
            ax1.set_xlim(max(xticks)+10, min(xticks)-10) 
            
            #ax2.set_ylim([2.5e-3, 6.0e-3])
            #ax2.set_yticks(np.arange(2.5e-3, 6.5e-3, 1e-3))
            ax2.tick_params(axis="y", labelsize=fontsize-10, colors=shapecolor)
            ax2.ticklabel_format(axis="y", style="sci", scilimits=(-3,-3))
            #ax2.yaxis.get_major_formatter().set_powerlimits((-1,2))
            ax2.set_ylabel('Input DS (Shape)', fontsize=fontsize)
            ax2.yaxis.offsetText.set_fontsize(fontsize-10)
            ax2.xaxis.offsetText.set_fontsize(fontsize-10)
            
            lns = lns1+lns2
            labs = [l.get_label() for l in lns]
            ax1.legend(lns,labs, fontsize=fontsize, loc="lower left")

            plt.tight_layout()
            save_path = f'cache/vc_angle_limit/train_{key}.png'
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

