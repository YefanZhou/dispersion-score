import glob
from os.path import join
import numpy as np
import sys
import matplotlib.pyplot as plt 
sys.path.append("../")
from eval_utils import mean_std
import argparse
TEST_SAMPLE_NUM = 8762
fontsize = 65
linewidth = 7
markersize = 30

def main(opt):
    base_dir = '../scripts/shapenet13_views_search/'
    pred_dir = 'checkpoints_pred'
    
    checkpoint_path = "stats_cache/nviews_search_avginertia.npy"
    if not opt.preload:
        data_path = glob.glob(join(base_dir, pred_dir, '*.npz'))
        data_path.sort()
        nviews = [1, 3, 6, 9, 12, 15, 18]  # 18 
        nviews_path_dic = {}
        for nv in nviews:
            pred_paths = glob.glob(join(base_dir, pred_dir, f'*trainnv{nv}_*.npz'))
            pred_paths.sort()
            assert len(pred_paths) == 3, "loading path error"
            nviews_path_dic[f'{nv}'] = pred_paths
        
        view_gt_path = '../scripts/shapenet13_benchmark/checkpoints_gt/viewer_test_points_rgb_RGB_mag0_n_op0_prob0_trainnv01_testnv01_8762.npz'
        #viewer_gt = np.load(view_gt_path, allow_pickle=True)
        
        save_path = 'cache/vc_nviews_search_3Seed.png'

        pred_stats_dic = {'AP_Inertia_k0p10':[], 
                    'AP_Inertia_k0p50':[], 
                    'KMedoids_Inertia_k100p0':[],
                    'KMedoids_Inertia_k500p0':[], 
                    'chamfer_stats':[], 
                    'fscore_stats':[]}
        for nv in nviews_path_dic:
            tmp_data_dic = {'AP_Inertia_k0p10':[], 
                            'AP_Inertia_k0p50':[], 
                            'KMedoids_Inertia_k100p0':[],
                            'KMedoids_Inertia_k500p0':[], 
                            'chamfer_stats':[], 
                            'fscore_stats':[]}
            nv_path_seedlist = nviews_path_dic[nv]
            print(nv_path_seedlist)
            for nv_path in nv_path_seedlist:
                stats_data = np.load(nv_path, allow_pickle=True)
                for key in tmp_data_dic:
                    if "Inertia" in key:
                        tmp_data_dic[key].append(stats_data[key][0]/TEST_SAMPLE_NUM)  
                    else:
                        tmp_data_dic[key].append(stats_data[key][0])


            print('KMedoids_Inertia_k500p0', tmp_data_dic['KMedoids_Inertia_k500p0'])
            print('chamfer_stats', tmp_data_dic['chamfer_stats'])
            print('fscore_stats', tmp_data_dic['fscore_stats'])
            print('----------------')

            assert len(tmp_data_dic[key]) == 3,f"{nv_path} data loading error"


            for key in tmp_data_dic:
                tmp_data_dic[key] = mean_std(tmp_data_dic[key])

            for key in tmp_data_dic:
                pred_stats_dic[key].append(tmp_data_dic[key])
            
        sys.exit(0)
        np.save(checkpoint_path, pred_stats_dic)
        
    else:
        recons_color = 'darkcyan'
        gt_color = 'red'
        chamfer_color = 'orange'
        fscore_color = 'blue'
        pred_stats_dic = np.load(checkpoint_path, allow_pickle=True)
        pred_stats_dic = pred_stats_dic.item()
        stats_dic = pred_stats_dic
        n_views = [1, 3, 6, 9, 12, 15, 18]
        #key_list = ['KMedoids_Inertia_k100p0', 'KMedoids_Inertia_k500p0'] 
        #key_list = ['chamfer_stats', 'fscore_stats']
        #key_list = ['AP_Inertia_k0p10', 'AP_Inertia_k0p50']
        key_list = ['KMedoids_Inertia_k500p0', 'chamfer_stats', 'fscore_stats']
        legend_dic = {'KMedoids_Inertia_k100p0': "n_clusters=100", 
                    'KMedoids_Inertia_k500p0': "n_clusters=500"}
        viewer_test_gt_points_path = "../scripts/shapenet13_benchmark/checkpoints_gt/viewer_test_points_rgb_RGB_mag0_n_op0_prob0_trainnv01_testnv01_8762.npz"
        viewer_test_gt_points_stats = np.load(viewer_test_gt_points_path)
        viewer_test_gt_points_inertia_mean = viewer_test_gt_points_stats[key_list[0]][0] / viewer_test_gt_points_stats['sample_num']
        
        xticks = n_views
        plt.style.use('ggplot')
        _, ax1 = plt.subplots(figsize=(18,14))
        ax1.set_xlabel(f"# of views per shape \n (# times more training samples)", fontsize=fontsize)
        ax1.set_ylabel('Output DS', fontsize=fontsize)
        inertia_list = stats_dic['KMedoids_Inertia_k500p0'] 
        ax1.plot(xticks, [item[0] for item in inertia_list], 'o-', c=recons_color, label="Reconstructed Shape", linewidth=linewidth, markersize=markersize)
        ax1.plot(xticks, len(xticks) * [viewer_test_gt_points_inertia_mean], '--', c=gt_color, linewidth=linewidth,)
        for x, y in zip(xticks, [item[0] for item in inertia_list]):
            ax1.text(x-1, y+0.0001, f"{y*1000:.2f}", color="black", fontsize=fontsize-20)
        ax1.fill_between(xticks, [item[0] - item[1] for item in inertia_list], [item[0] + item[1] for item in inertia_list], alpha=0.2, facecolor=recons_color)
        #ax1.set_ylim([4.0e-3, 1.0e-2])
        #ax1.ticklabel_format(axis="y", style="sci", scilimits=(-1,1))   
        ax1.yaxis.get_major_formatter().set_powerlimits((-3,-3))
        ax1.yaxis.offsetText.set_fontsize(fontsize-10)
        ax1.xaxis.offsetText.set_fontsize(fontsize-10)
        ax1.set_xticks(xticks)
        ax1.tick_params(axis="x", labelsize=fontsize-10)
        ax1.tick_params(axis="y", labelsize=fontsize-10)
        
        ax1.legend(['Reconstruction', 'GT'], loc='center left', fontsize=fontsize)
        plt.tight_layout()
        save_path = join('cache/Viewsearch', f"nviews_inertia.png")
        plt.savefig(save_path, bbox_inches='tight')
        
        xticks = n_views
        _, ax1 = plt.subplots(figsize=(18,14))
        plt.style.use('ggplot')
        ax1.set_xlabel(f"# of views per shape \n (# times more training samples)", fontsize=fontsize)
        ax1.set_ylabel('CD', fontsize=fontsize)
        ax2 = ax1.twinx()
        chamfer_list = stats_dic['chamfer_stats'] 
        fscore_list = stats_dic['fscore_stats'] 
        lns1 = ax1.plot(xticks, [item[0] for item in chamfer_list], 
                'o-', c=chamfer_color, 
                linewidth=linewidth, markersize=markersize, label="CD")
        ax1.fill_between(xticks, [item[0] - item[1] for item in chamfer_list], [item[0] + item[1] for item in chamfer_list], alpha=0.2, facecolor=chamfer_color)
        ax1.ticklabel_format(axis="y", style="sci", scilimits=(-1,1))   
        ax1.yaxis.get_major_formatter().set_powerlimits((0,0))
        ax1.yaxis.offsetText.set_fontsize(fontsize-10)
        ax1.xaxis.offsetText.set_fontsize(fontsize-10)
        ax1.set_xticks(xticks)
        ax1.tick_params(axis="y", labelsize=fontsize-10, colors=chamfer_color)
        ax1.tick_params(axis="x", labelsize=fontsize-10)
        
        lns2 = ax2.plot(xticks, [item[0] for item in fscore_list], 
                'o-', c=fscore_color, 
                linewidth=linewidth, markersize=markersize, label="F-score")
        ax2.fill_between(xticks, [item[0] - item[1] for item in fscore_list], [item[0] + item[1] for item in fscore_list], alpha=0.2, facecolor=fscore_color)

        ax2.tick_params(axis="y", labelsize=fontsize-10, colors=fscore_color)
        ax2.tick_params(axis="x", labelsize=fontsize-10)
        
        ax2.set_ylabel('F-score', fontsize=fontsize)
        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, fontsize=fontsize, loc='center right')
        plt.tight_layout()
        save_path = join('cache/Viewsearch', f"nviews_chamferFscore.png")
        plt.savefig(save_path, bbox_inches='tight')




    #print(pred_stats_dic)

        
    # plt.figure(figsize=(15, 8))
    # plt.style.use('ggplot')
    # xticks = nviews
    
    # for idx, key in enumerate(data_dic):
    #     plt.subplot(2, 4, idx + 1)
    #     plt.plot(xticks, data_dic[key], 'o-', c=clr_list[idx], label=key)
        
    #     #plt.ylim(min(data_dic[key]), max(data_dic[key]))
    #     if key not in ['chamfer_stats', 'fscore_stats']:
    #         #print(object_gt[key][0], viewer_gt[key][0])
    #         #plt.scatter([xticks[0]], object_gt[key][0], c='k', label='GT')
    #         #plt.scatter([xticks[1]], viewer_gt[key][0], c='k', label='GT')
    #         plt.title("Pred Points Clustering Extent")
    #         plt.plot(xticks, len(xticks) * [viewer_gt[key][0]], c='k', label='GT')
    #     else:
    #         plt.title("Reconstruction")

    #     plt.xticks(xticks, fontsize=10)
    #     if idx >= 4:
    #         plt.xlabel("Number of Views")
    #     plt.legend()

    # plt.suptitle("VC num of views search (8762/8762) ")
    # plt.tight_layout()
    #plt.savefig(save_path)
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser("")
    parser.add_argument("--preload", action='store_true', default=False)
    parser.add_argument("--subplot", action='store_true', default=False)
    parser.add_argument("--onefigure", action='store_true', default=False)
    opt = parser.parse_args()
    main(opt)