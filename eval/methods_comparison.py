"""This script is used to verity the correctness of Clustering method from What3D tatacheko
"""
import os
import sys
sys.path.append("../")
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from os.path import join
import random
import auxiliary.my_utils as my_utils

filelst_path = '../dataset/data/filelists/test.lst'
rendering_path = '../dataset/data/ShapeNetV1RenderingsLocal'
viewer_cluster_basedir = "../log/benchmark/20210202_2055_viewer_clustering_seed1"
object_cluster_basedir = "../log/benchmark/20210203_1048_object_clustering_seed1"
object_oraclenn_basedir =  "../log/benchmark/20210203_0307_object_oraclenn_seed1"
viewer_oraclenn_basedir = "../log/benchmark/20210203_0651_viewer_oraclenn_seed1"

object_atlassph_basedir = "../log/benchmark/20210203_0333_object_atlasnetSPHERE1_seed1"
viewer_atlassph_basedir = "../log/benchmark/20210130_0505_viewer_atlasnetSPHERE1_seed1"

with open(filelst_path, 'r') as f:
    id_lsts = f.read().splitlines()

#obj_matrix = np.load(join(object_cluster_basedir, "object_trainnv1_cls13True.npy"))
#obj_res = np.load(join(object_cluster_basedir, "cluster_res_500.npy"), allow_pickle=True)
#obj_res = obj_res.item()

#viewer_matrix = np.load(join(viewer_cluster_basedir, "viewer_trainnv1_cls13True.npy"))
#viewer_res = np.load(join(viewer_cluster_basedir, "cluster_res_500.npy"), allow_pickle=True)
#viewer_res = viewer_res.item()


# kmedoids = KMedoids(n_clusters=500, 
#         random_state=1, metric='precomputed', init='k-medoids++').fit(obj_matrix)
# score = kmedoids.inertia_
# print(score/obj_matrix.shape[0])


# kmedoids = KMedoids(n_clusters=500, 
#         random_state=1, metric='precomputed', init='k-medoids++').fit(viewer_matrix)
# score = kmedoids.inertia_
# print(score/viewer_matrix.shape[0])

random.seed(0)
##OC vis
obj_pred_cluster = np.load(join(object_cluster_basedir, '210205_0722_object_optimize1_pretrainedTrue_decay30', "prediction.npy"))
obj_gt_cluster = np.load(join(object_cluster_basedir, '210205_0722_object_optimize1_pretrainedTrue_decay30', "gt.npy"))
obj_pred_oraclenn = np.load(join(object_oraclenn_basedir, "prediction.npy"))
obj_pred_atlassph = np.load(join(object_atlassph_basedir, "prediction.npy"))
# #viewer_pred_cluster = np.load(join(object_cluster_basedir, '210205_0722_object_optimize1_pretrainedTrue_decay30', "prediction.npy"))
# print(obj_gt_cluster.shape)

# for i in random.sample(range(obj_gt_cluster.shape[0]), 10):
#     pred_gt = np.concatenate((obj_gt_cluster[[i]], obj_pred_cluster[[i]], obj_pred_oraclenn[[i]]), axis=0)
#     print(pred_gt.shape)
#     my_utils.visuaize_pts(pred_gt, "GT                                        Clustering                                         Oracle NN")
#     plt.tight_layout()
#     plt.show()

##VC vis
viewer_pred_cluster = np.load(join(viewer_cluster_basedir, '210219_0200_viewer_clustering_optimize1_pretrainedTrue_decay30_Seed2_cluster_res_500', "prediction.npy"))
viewer_gt_cluster = np.load(join(viewer_cluster_basedir, '210219_0200_viewer_clustering_optimize1_pretrainedTrue_decay30_Seed2_cluster_res_500', "gt.npy"))
viewer_pred_oraclenn = np.load(join(viewer_oraclenn_basedir, "prediction.npy"))
viewer_pred_atlassph = np.load(join(viewer_atlassph_basedir, "prediction.npy"))                        

# for i in random.sample(range(viewer_gt_cluster.shape[0]), 20):
#     pred_gt = np.concatenate((viewer_gt_cluster[[i]], viewer_pred_cluster[[i]]), axis=0)
#     print(pred_gt.shape)
#     my_utils.visuaize_pts(pred_gt, "GT                                        Clustering", elev=0, azim=0)
#     plt.tight_layout()
#     plt.show()

model_names = ['GT', 'Clustering', 'AtlasNetSph', 'OracleNN']
##VC OC vis
for i in random.sample(range(viewer_gt_cluster.shape[0]), 20):
    img_path = join(rendering_path, id_lsts[i], "rendering/00.png")
    rendering = cv2.imread(img_path)
    #fig = plt.figure()
    #plt.imshow(rendering)
    #plt.tight_layout()
    #plt.show()
    point_idx = random.sample(range(obj_pred_atlassph.shape[1]), 2500)
    obj_pred_gt = np.concatenate((obj_gt_cluster[[i]], obj_pred_cluster[[i]], obj_pred_atlassph[[i]][:,point_idx,:], obj_pred_oraclenn[[i]]), axis=0)
    view_pred_gt = np.concatenate((viewer_gt_cluster[[i]], viewer_pred_cluster[[i]], viewer_pred_atlassph[[i]][:,point_idx,:], viewer_pred_oraclenn[[i]]), axis=0)
    print(obj_pred_gt.shape)
    
    rows = 2
    cols = max(obj_pred_gt.shape[0], view_pred_gt.shape[0])  + 1
    
    fig = plt.figure(figsize=(3 * cols, 3 * rows))
    ax = fig.add_subplot(rows, cols, 1)
    plt.imshow(rendering)

    elev, azim = 30, 135
    for idx in range(0, obj_pred_gt.shape[0]):
        print(idx)
        sample = obj_pred_gt[idx]
        ax = fig.add_subplot(rows, cols, idx + 2, projection='3d')
        ax.set_title(model_names[idx])
        ax.scatter(sample[:,2], sample[:,0], sample[:,1], s= 5, alpha=0.5)
        ax.set_xlim([-0.6,0.6])
        ax.set_ylim([-0.6,0.6])
        ax.set_zlim([-0.6,0.6])
        ax.set_xlabel('Z')
        ax.set_ylabel('X')
        ax.set_zlabel('Y')
        ax.view_init(elev, azim)
        ax.axis('off')

    elev, azim = 0, 0 
    for idx in range(0, view_pred_gt.shape[0]):
        print(idx)
        sample = view_pred_gt[idx]
        ax = fig.add_subplot(rows, cols, cols+idx + 2, projection='3d')
        ax.set_title(model_names[idx])
        ax.scatter(sample[:,2], sample[:,0], sample[:,1], s= 5, alpha=0.5)
        ax.set_xlim([-0.6,0.6])
        ax.set_ylim([-0.6,0.6])
        ax.set_zlim([-0.6,0.6])
        ax.set_xlabel('Z')
        ax.set_ylabel('X')
        ax.set_zlabel('Y')
        ax.view_init(elev, azim)
        ax.axis('off')

    plt.tight_layout()
    plt.show()



