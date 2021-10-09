import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import argparse
sys.path.append("../")
from eval_utils import mean_std
parser = argparse.ArgumentParser()


def main(opt):
    TEST_SAMPLE_NUM = 8762
    mode_list = ['object', 'viewer']
    clr_list = ['b', 'r', 'g', 'y', 'r']
    save_path = 'cache/oc_vc_bar_reconstruction.png'
    cached_save_path = join('stats_cache', 'oc_vc_avginertia.npy')

    key_lst = ['KMedoids_Inertia_k100p0', 'KMedoids_Inertia_k500p0']
    #key_lst =  ['chamfer_stats', 'fscore_stats']
    legend_dic = {'KMedoids_Inertia_k100p0': "n_clusters=100", 
                'KMedoids_Inertia_k500p0': "n_clusters=500"}
    if not opt.preload:
        base_dir = '../scripts/shapenet13_oc_vc/'
        pred_dir = 'checkpoints_pred'

        data_path = glob.glob(join(base_dir, pred_dir, '*.npz'))
        data_path.sort()
        
        data_path = {'object': [s for s in data_path if "object" in s], 
                'viewer': [s for s in data_path if "viewer" in s]}

        abb_dic = { 'psgn':'PSGN', 
                    'foldnetSQUARE1':'FoldNet', 
                    'atlasnetSPHERE1': 'AtlasNetSphere1', 
                    'atlasnetSQUARE25': 'AtlasNetSquare25',
                    'optimize1': 'Cluster', 
                    'oraclenn': 'OracleNN'}

        data_dic = {'AP_Inertia_k0p10':{}, 
                    'AP_Inertia_k0p50':{}, 
                    'KMedoids_Inertia_k50p0':{},
                    'KMedoids_Inertia_k100p0':{},
                    'KMedoids_Inertia_k500p0':{}, 
                    'chamfer_stats':{}, 
                    'fscore_stats':{}}
        
        print(data_path)
        for mode in mode_list:
            mode_path = data_path[mode]
            for seed_path in mode_path:
                data = np.load(seed_path, allow_pickle=True)
                for key in data_dic:
                    if mode not in data_dic[key]:
                        data_dic[key].update({mode: []})
                    if "Inertia" in key:
                        assert data['sample_num'] == TEST_SAMPLE_NUM
                        data_dic[key][mode].append(data[key][0]/TEST_SAMPLE_NUM)
                    else:
                        data_dic[key][mode].append(data[key][0])
        
        print(data_dic['KMedoids_Inertia_k500p0'])
        for key in data_dic:
            for mode in data_dic[key]:
                data_dic[key][mode] = mean_std(data_dic[key][mode])

        print(data_dic['KMedoids_Inertia_k500p0'])
        #np.save(cached_save_path, data_dic)

    else:
        fontsize = 60
        color = 'darkcyan'
        data_dic = np.load(cached_save_path, allow_pickle=True)
        oc_test_gt = np.load("../scripts/shapenet13_benchmark/checkpoints_gt/object_test_points_rgb_RGB_mag0_n_op0_prob0_trainnv01_testnv01_8762.npz")
        vc_test_gt = np.load("../scripts/shapenet13_benchmark/checkpoints_gt/viewer_test_points_rgb_RGB_mag0_n_op0_prob0_trainnv01_testnv01_8762.npz")
        TEST_SAMPLE_NUM = oc_test_gt['sample_num']
        data_dic = data_dic.item()
        oc_pred_inertia_avg = data_dic['KMedoids_Inertia_k500p0']['object']
        vc_pred_inertia_avg = data_dic['KMedoids_Inertia_k500p0']['viewer'] 

        print(oc_pred_inertia_avg, vc_pred_inertia_avg)
        oc_test_gt_inertia = oc_test_gt['KMedoids_Inertia_k500p0'] / TEST_SAMPLE_NUM
        vc_test_gt_inertia = vc_test_gt['KMedoids_Inertia_k500p0'] / TEST_SAMPLE_NUM

        plt.style.use('ggplot')
        xticks = ['OC', 'VC']
        _, ax1 = plt.subplots(figsize=(18,14))
        width = 0.2 
        ax1.bar(np.array([0.4]), [oc_pred_inertia_avg[0]], yerr=[oc_pred_inertia_avg[1]], 
        align='center', alpha=1, color=color, width=width, ecolor='black', capsize=100)
        ax1.text(x=0.36 , y=oc_pred_inertia_avg[0] * 1.1, s=f"{oc_pred_inertia_avg[0]*1000:.2f}", fontsize=fontsize-10)
        ax1.set_xlabel("Shape Coordinate Representation", fontsize=fontsize, labelpad=20)
        ax1.bar(np.array([0.8]), [vc_pred_inertia_avg[0]], yerr=[vc_pred_inertia_avg[1]], 
        align='center', alpha=1, color=color, width=width, ecolor='black', capsize=100)
        ax1.text(x=0.76 , y=vc_pred_inertia_avg[0] * 1.30, s=f"{vc_pred_inertia_avg[0]*1000:.2f}", fontsize=fontsize-10)
        ax1.text(x=0.70 , y=vc_pred_inertia_avg[0] * 1.05, s=f"(More recons.)", fontsize=fontsize-10)
        plt.xticks([0.4, 0.8], xticks)
        ax1.legend(['Reconstruction'], fontsize=fontsize, loc="upper left")
        ax1.set_yticks(np.arange(0, 2.4e-2, 0.6e-2))
        ax1.tick_params(axis="y", labelsize=fontsize-10)
        ax1.tick_params(axis="x", labelsize=fontsize-10)
        ax1.ticklabel_format(axis="y", style="sci", scilimits=(-3,-3))
        
        ax1.set_ylabel('Output DS', fontsize=fontsize)
        ax1.yaxis.offsetText.set_fontsize(fontsize-10)
        ax1.xaxis.offsetText.set_fontsize(fontsize-10)
        ax1.set_ylim([0, 2.4e-2])

        plt.tight_layout()
        
        save_path = 'cache/MoreShapeAug/oc_vc_pred.png'
        plt.savefig(save_path, bbox_inches='tight')
        

if __name__ == '__main__':
    parser.add_argument("--preload", action='store_true', default=False)
    parser.add_argument("--multibar", action='store_true', default=False)
    opt = parser.parse_args()
    
    main(opt)
