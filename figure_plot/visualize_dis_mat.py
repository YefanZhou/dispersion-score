import os
import sys
import argparse
import numpy as np
from os.path import join
from pathlib import Path
import matplotlib.pyplot as plt
import glob

##################GT Trainset Image Points   ###################################
##############################################################################

base_dir = 'eval/eval_results/eval_ds_synthetic/checkpoints_1'
clsize_lst   = [1, 5, 10, 15, 20, 25, 30, 35]

clsize_lst.reverse()
save_dir = 'figure_plot/figures/synthetic/vis_dismat/TrainImage/ImageAug'
Path(save_dir).mkdir(parents=True, exist_ok=True)
for clsize in clsize_lst:
    npy_path = join(base_dir, f"dismat_object_train_image_cluster_image_aug_2by10_cltsize{clsize:02}_numsample20.npy")
    dm = np.load(npy_path)
    plt.figure(figsize=(10,10))
    plt.imshow(dm, interpolation='nearest', cmap=plt.cm.get_cmap("jet"))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(join(save_dir,f"dismat_object_train_image_cluster_image_aug_2by10_cltsize{clsize:02}_numsample20.png"), bbox_inches='tight')
    plt.close()

save_dir = 'figure_plot/figures/synthetic/vis_dismat/TrainPoints/ImageAug'
Path(save_dir).mkdir(parents=True, exist_ok=True)
for clsize in clsize_lst:
    npy_path = join(base_dir, f"dismat_object_train_points_cluster_image_aug_2by10_cltsize{clsize:02}_numsample20.npy")
    dm = np.load(npy_path)
    plt.figure(figsize=(10,10))
    plt.imshow(dm, interpolation='nearest', cmap=plt.cm.get_cmap("jet"))
    plt.axis("off") 
    plt.tight_layout()
    plt.savefig(join(save_dir,f"dismat_object_train_points_cluster_image_aug_2by10_cltsize{clsize:02}_numsample20.png"), bbox_inches='tight')
    plt.close()

save_dir = 'figure_plot/figures/synthetic/vis_dismat/TrainImage/ShapeAug'
Path(save_dir).mkdir(parents=True, exist_ok=True)
for clsize in clsize_lst:
    npy_path = join(base_dir, f"dismat_object_train_image_cluster_shape_aug_2by10_cltsize{clsize:02}_numsample20.npy")
    dm = np.load(npy_path)
    plt.figure(figsize=(10,10))
    plt.imshow(dm, interpolation='nearest', cmap=plt.cm.get_cmap("jet"))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(join(save_dir, f"dismat_object_train_image_cluster_shape_aug_2by10_cltsize{clsize:02}_numsample20.png"), bbox_inches='tight')
    plt.close()

save_dir = 'figure_plot/figures/synthetic/vis_dismat/TrainPoints/ShapeAug'
Path(save_dir).mkdir(parents=True, exist_ok=True)
for clsize in clsize_lst:
    npy_path = join(base_dir, f"dismat_object_train_points_cluster_shape_aug_2by10_cltsize{clsize:02}_numsample20.npy")
    dm = np.load(npy_path)
    plt.figure(figsize=(10,10))
    plt.imshow(dm, interpolation='nearest', cmap=plt.cm.get_cmap("jet"))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(join(save_dir, f"dismat_object_train_points_cluster_shape_aug_2by10_cltsize{clsize:02}_numsample20.png"), bbox_inches='tight')
    plt.close()

#########################Pred Points###########################
###############################################################
base_dir = 'eval/eval_results/eval_ds_synthetic/checkpoints_pred'

save_dir = 'figure_plot/figures/synthetic/vis_dismat/Prediction/ImageAug'
Path(save_dir).mkdir(parents=True, exist_ok=True)
clspath_list = glob.glob(join(base_dir, "*_image_aug_*"))
clspath_list.sort(reverse=True)

for clspath in clspath_list:
    cls_stats = np.load(clspath, allow_pickle=True)
    dm = cls_stats['1'].item()['dm']
    plt.figure(figsize=(10,10))
    plt.imshow(dm, interpolation='nearest', cmap=plt.cm.get_cmap("jet")) #
    plt.axis("off")
    plt.tight_layout()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
        hspace = 0, wspace = 0)
    plt.margins(0,0)     
    plt.savefig(join(save_dir, clspath.split('/')[-1].replace('npz', 'png')), bbox_inches='tight')
    #plt.show()
    plt.close()


save_dir = 'figure_plot/figures/synthetic/vis_dismat/Prediction/ShapeAug'
Path(save_dir).mkdir(parents=True, exist_ok=True)
clspath_list = glob.glob(join(base_dir, "*_shape_aug_*"))
clspath_list.sort(reverse=True)

for clspath in clspath_list:
    cls_stats = np.load(clspath, allow_pickle=True)
    dm = cls_stats['1'].item()['dm']
    plt.figure(figsize=(10,10))
    plt.imshow(dm, interpolation='nearest', cmap=plt.cm.get_cmap("jet")) #
    plt.axis("off")
    plt.tight_layout()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
        hspace = 0, wspace = 0)
    plt.margins(0,0)  
    plt.savefig(join(save_dir, clspath.split('/')[-1].replace('npz', 'png')), bbox_inches='tight')
    plt.close()

