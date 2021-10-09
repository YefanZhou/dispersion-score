"""Compute distance matrix of clusters data, plot the T-SNE, and compute K-medoids inertia
    in differt number of clusters 
    The standard deviation of the clusters.
    https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
    https://en.wikipedia.org/wiki/Elbow_method_(clustering) elbow 
    The method can be traced to speculation by Robert L. Thorndike in 1953.[1]
"""
import glob
import argparse
import os
import sys
import numpy as np
from os.path import join
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
sys.path.append('../../')
from eval.metric import cluster_eval
import auxiliary.my_utils as my_utils
from sklearn.manifold import TSNE



color_list = ['orange', 'blue', 'red', 'green', 'brown', 'magenta', 'purple', 'darkcyan']
tsne_color_list = ['b', 'g', 'r', 'c', 'm','y', 'k', 'pink', 'brown', 'cyan']
parser = argparse.ArgumentParser(
    description='sum the integers at the command line')
parser.add_argument(
    '--save_folder', type=str, default="clusters_data",
    help='')
parser.add_argument(
    '--seed', type=int, default=1,
    help='')
parser.add_argument(
    '--tsne_plot', action='store_true', default=False,
    help='')

args = parser.parse_args()

my_utils.plant_seeds(args.seed)

npz_fpath = glob.glob(join(args.save_folder, "*.npz"))
npz_fpath.sort()
tsne = TSNE(n_components=2, metric='precomputed', random_state=args.seed)
clustering_method = "KMedoids"
eval_method = "Inertia"
inertia_dic = {}   # first level key is std,  # second level key is number of clusters

nclusters_list = list(range(2, 102, 2))
print(nclusters_list)

inertia_dic_path = join(args.save_folder, f"inertia_dic_max{max(nclusters_list)}_min{min(nclusters_list)}_step{nclusters_list[1]-nclusters_list[0]}.npy")
print(inertia_dic_path)
if os.path.exists(inertia_dic_path) and not args.tsne_plot:
    inertia_dic = np.load(inertia_dic_path, allow_pickle=True)
    inertia_dic = inertia_dic.item()
else:
    print("Plot t-SNE")
    for idx in range(len(npz_fpath)):
        file_name = npz_fpath[idx]
        cluster_std = int(file_name[file_name.find("std") + 3:file_name.find("std") + 4])
        inertia_dic.update({cluster_std:[]})
        data_dic = np.load(file_name)
        data = data_dic['data']
        label = data_dic['label']
        num_of_clusters = max(label) + 1
        
        # for i in range(num_of_clusters):
        #     plt.scatter(data[label == i, 0], data[label == i, 1], color=color_list[i], s=10)
        # plt.title(f"2D data std:{cluster_std}")
        # plt.tight_layout()
        # plt.show()
        cond_matrix = pdist(data, 'sqeuclidean')
        squa_matrix = squareform(cond_matrix)

        ## Verify if squared distance matrix correct
        # i1 = np.random.randint(0, squa_matrix.shape[0])
        # i2 = np.random.randint(0, squa_matrix.shape[0])
        # print((data[i1][0] - data[i2][0]) ** 2 + (data[i1][1] - data[i2][1]) ** 2)
        # print(squa_matrix[i1, i2])
        # print(squa_matrix[i2, i1])
        # print("--------")

        if args.tsne_plot:
            embeddings = tsne.fit_transform(squa_matrix)

            #plt.figure(figsize=(3,3))
            plt.figure(figsize=(5,5))
            vis_x = embeddings[:, 0]
            vis_y = embeddings[:, 1]
            #plt.title(f"t-SNE std:{cluster_std}")
            c_list = []
            for label_idx in label:
                c_list.append(tsne_color_list[label_idx])
            plt.scatter(vis_x, vis_y, c=c_list, s=80)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(join(args.save_folder, f"tsne_{file_name.split('/')[-1].replace('npz', 'png')}"), bbox_inches="tight")
            #plt.show()
        else:
            print("Compute KM")
            for n_cluster in nclusters_list:
                inertia, _ = cluster_eval(c_method=clustering_method, e_method=eval_method,
                            distance_matrix=squa_matrix, n_cluster=n_cluster, 
                            pc=-1, mean=False)
                SAMPLE_NUM = squa_matrix.shape[0] 
                
                inertia_dic[cluster_std].append(inertia/SAMPLE_NUM)
    
        print("SAMPLE NUM", SAMPLE_NUM)
        np.save(inertia_dic_path, inertia_dic)


plt.figure(figsize=(15, 10))
for idx, cluster_std in enumerate(inertia_dic):
    plt.plot(nclusters_list, inertia_dic[cluster_std], c=color_list[idx], label=f'cluster std={cluster_std}', linewidth=4)

plt.ylabel("Dispersion Score", fontsize=35, labelpad=20)
plt.xlabel("number of clusters", fontsize=35, labelpad=20)

plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.legend(fontsize=25)
#plt.show()
plt.tight_layout()
plt.savefig(join(args.save_folder, f"km_inertia_max{max(nclusters_list)}_min{min(nclusters_list)}_step{nclusters_list[1]-nclusters_list[0]}.png"))

    

