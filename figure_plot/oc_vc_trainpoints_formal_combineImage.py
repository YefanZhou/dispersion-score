import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import argparse
sys.path.append("../")
from eval_utils import mean_std
parser = argparse.ArgumentParser()

fontsize = 60
linewidth = 7
markersize = 30
imagecolor = 'goldenrod'
shapecolor = 'mediumblue'
TRAIN_SAMPLE_SIZE = 35021
def main(opt):
    mode_list = ['object-centered', 'viewer-centered']
    clr_list = ['b', 'r', 'g', 'y', 'r']
    cache_save_fname = 'oc_vc_trainpoints.npy'
    
    legend_dic = {'KMedoids_Inertia_k100p0': "n_clusters=100", 
                'KMedoids_Inertia_k500p0': "n_clusters=500"}
    data_dic = {}
    
    if not opt.preload:
        data_dic = {}
        for key in key_lst:
            data_dic.update({key:{}})
        for idx, mode in enumerate(['object', 'viewer']):
            path = join(base_dir, f'{mode}_train_points_trainnv01_testnv01_35021.npz')
            stats = np.load(path, allow_pickle=True)
            for key in key_lst:
                data_dic[key].update({mode:(stats[key][0], stats[key][1])})
        
        

            # for key in key_lst:
            #     data_dic.update({key:[]})

        np.save(join('stats_cache', cache_save_fname), data_dic)

    else:
        key_lst = ['KMedoids_Inertia_k500p0']
        #data_dic = np.load(join('stats_cache', cache_save_fname), allow_pickle=True)

        oc_trainpoints = np.load("../scripts/shapenet13_benchmark/checkpoints_gt/object_train_points_trainnv01_testnv01_35021.npz")
        oc_shape_inertia_meanstd = oc_trainpoints[key_lst[0]]
        oc_shape_inertia = oc_shape_inertia_meanstd[0]/TRAIN_SAMPLE_SIZE

        vc_trainpoints = np.load("../scripts/shapenet13_benchmark/checkpoints_gt/viewer_train_points_trainnv01_testnv01_35021.npz")
        vc_shape_inertia_meanstd = vc_trainpoints[key_lst[0]]
        vc_shape_inertia = vc_shape_inertia_meanstd[0]/TRAIN_SAMPLE_SIZE
        
        image_stats_dic = np.load('stats_cache/image_augment_geometric_AllImageInertia_AverageInertia.npy', allow_pickle=True)
        image_stats_dic = image_stats_dic.item()

        train_image_stats = np.load('../scripts/shapenet13_img_aug_params/checkpoints_1/submission/object_train_image_geometricaugment_None_mag0_magnitude0_n_op0_prob0_trainnv01_testnv01_35021.npz')
        origin_image_inertia = image_stats_dic[key_lst[0]][0]
        assert train_image_stats[key_lst[0]][0]/TRAIN_SAMPLE_SIZE == origin_image_inertia[0]
        print(image_stats_dic, origin_image_inertia)
        print(oc_shape_inertia, vc_shape_inertia)

        plt.style.use('ggplot')
        #xticks = mode_list
        xticks = ['OC', 'VC']
        _, ax1 = plt.subplots(figsize=(18,14))
        ax2 = ax1.twinx()

        width = 0.2
        height = 0.1
        ax1.set_xlabel("Shape Coordinate Representation", fontsize=fontsize, labelpad=20)
        ax1.bar(np.array([0.2]), [origin_image_inertia[0]], 
        align='center', width=width, color=imagecolor, ecolor='black', label='Training Image', capsize=10)  
        ax1.text(x=0.13 , y=origin_image_inertia[0] + 2, s=f"{origin_image_inertia[0]:.2f}", fontsize=fontsize-10)
        ax1.set_yticks(np.arange(0, 300, 50))

        ax2.bar(np.array([0.4]), [oc_shape_inertia], 
        align='center',  width=width, color=shapecolor, ecolor='black', label='Training Shape', capsize=10)
        ax2.text(x=0.34 , y=oc_shape_inertia * 1.05, s=f"{oc_shape_inertia*1000:.2f}", fontsize=fontsize-10)
        
        ax1.legend(['Training Image'], fontsize=fontsize, loc="upper left")
        ax2.legend(['Training Shape'], fontsize=fontsize, loc="upper left", bbox_to_anchor=(0, 0.9))
        
        ax1.bar(np.array([0.8]), [origin_image_inertia[0]], 
        align='center', width=width, color=imagecolor, ecolor='black', capsize=10)
        ax1.text(x=0.73 , y=origin_image_inertia[0] + 2, s=f"{origin_image_inertia[0]:.2f}", fontsize=fontsize-10)
        ax2.bar(np.array([1.0]), [vc_shape_inertia],
        align='center',  width=width, color=shapecolor, ecolor='black', capsize=10)
        ax2.text(x=0.92 , y=vc_shape_inertia * 1.005, s=f"{vc_shape_inertia*1000:.2f}", fontsize=fontsize-10)
        
        ax1.tick_params(axis="x", labelsize=fontsize-10)
        ax1.tick_params(axis="y", labelsize=fontsize-10, colors=imagecolor)
        ax1.set_ylabel('Input DS (Image)', fontsize=fontsize)
        ax1.set_ylim([0, 350])
        
        ax2.set_yticks(np.arange(0, 3.0e-2, 0.6e-2))
        ax2.tick_params(axis="y", labelsize=fontsize-10, colors=shapecolor)
        ax2.ticklabel_format(axis="y", style="sci", scilimits=(-3,-3))
        
        ax2.set_ylabel('Input DS (Shape)', fontsize=fontsize)
        ax2.yaxis.offsetText.set_fontsize(fontsize-10)
        ax2.xaxis.offsetText.set_fontsize(fontsize-10)
        ax2.set_ylim([0, 3.5e-2])
        
        #lns = lns1+lns2
        #labs = [l.get_label() for l in lns]

        
        plt.xticks([0.3, 0.9], xticks, fontsize=fontsize)
        
        
        plt.tight_layout()
        save_path = 'cache/MoreShapeAug/oc_vc_TrainPointImage_Bar.png'
        plt.savefig(save_path, bbox_inches='tight')
        


if __name__ == '__main__':
    parser.add_argument("--preload", action='store_true', default=False)
    parser.add_argument("--multibar", action='store_true', default=False)
    opt = parser.parse_args()

    main(opt)
