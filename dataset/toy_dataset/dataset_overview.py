"""This scripts is used to test cubesphere dataset:
    1. If train, val, test overlap
    2. Rename image augment subsampled dataset, 
    3. check if the data consistent with data labeled with January (this one is May)
    4. check image augmentation train subsample json distribution and make distribution plot 
    5. generate shape augmentation train subsample json distribution and make distribution plots
"""
import sys
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from os.path import join
sys.path.append("../../")
import matplotlib as mpl
from auxiliary.my_utils import visuaize_pts, save_image, image_grid
# 2 x 10 json used in experiment before  1 5 10 15 20 25 30 35 (25 removed)

vis_folder= 'vis'
actual_shape_key_lst = []
local_base_path = '../../../data/cubesphere_1000'
new_sampled_point_cloud_name = 'pt_30000.npy'
train_json = "train_interp_1000.json"
test_json = "test_interp_1000.json"
val_json = "val_interp_1000.json"
fontsize = 85
linewidth = 7
markersize = 2000

with open(join(local_base_path, train_json), 'r') as f:
    train_lst = json.load(f)

with open(join(local_base_path, test_json), 'r') as f:
    test_lst = json.load(f)

with open(join(local_base_path, val_json), 'r') as f:
    val_lst = json.load(f)

print(f"Train Size:{len(train_lst)}, Val Size: {len(val_lst)}, Test Size: {len(test_lst)}")

## Test If OverLap
train_dir_lst = [train_lst[i]['dir_path'] for i in range(len(train_lst))]
test_dir_lst = [test_lst[i]['dir_path'] for i in range(len(test_lst))]
val_dir_lst = [val_lst[i]['dir_path'] for i in range(len(val_lst))]

actual_shape_key_lst += [train_lst[i]['shapekey_value'] for i in range(len(train_lst))]
actual_shape_key_lst += [test_lst[i]['shapekey_value'] for i in range(len(test_lst))]
actual_shape_key_lst += [val_lst[i]['shapekey_value'] for i in range(len(val_lst))]
actual_shape_key_lst.sort()

no_overlap = True
for train_dir in train_dir_lst:
    if train_dir in test_dir_lst:
        print(f"{train_dir} overlap in test")
        no_overlap = False
    if train_dir in val_dir_lst:
        print(f"{train_dir} overlap in val")
        no_overlap = False

for val_dir in val_dir_lst:
    if val_dir in test_dir_lst:
        no_overlap = False
        print(f"{val_dir} overlap in test")
print(f"No Overlap: {no_overlap}")


#Rename image clutter json 
def rename_imageaug_clutter_json(test_dir_lst):
    cluster_size_lst = [35, 30, 25, 20, 15, 10, 5, 1]
    for idx, cluster_size in enumerate(cluster_size_lst):
        clutter_path = join(local_base_path, f"cluster_samept_2by10_{cluster_size}.json")
        rename_clutter_path = join(local_base_path, f"cluster_image_aug_2by10_cltsize{cluster_size:02}.json")

        with open(clutter_path, 'r') as f:
            clutter_lst = json.load(f)

        ##Check clutter overlap OverLap
        clutter_dir_lst = [clutter_lst[i]['dir_path'] for i in range(len(clutter_lst))]
        no_overlap = True
        for clutter_dir in clutter_dir_lst:
            if clutter_dir in test_dir_lst:
                no_overlap = False
        
        
        for i in range(len(clutter_lst)):
            print(clutter_lst[i]['isometric_path'])
            print(clutter_lst[i]['ptcloud_path'])
            print("Before", clutter_lst[i]['model_path'])
            clutter_lst[i]['model_path'] = join(*clutter_lst[i]['ptcloud_path'].split('/')[:-1], 'model.obj')
            print("After", clutter_lst[i]['model_path'])
            print('---------------------')
        
        with open(rename_clutter_path, 'w', encoding='utf-8') as f:
            json.dump(clutter_lst, f, ensure_ascii=False, indent=4)

        print(f"Cluster Size: {cluster_size}, no_overlap: {no_overlap}")


def plot_imageaug_cluster(local_base_path):
    cluster_size_lst = [35, 30, 25, 20, 15, 10, 5, 1]
    total_scale = 1000.0
    xticks_symbol_list = []
    xticks_list = []
    ##Plot image clutter json
    plt.style.use('ggplot')
    _, ax1 = plt.subplots(figsize=(18,15))
    for idx, cluster_size in enumerate(cluster_size_lst):
        clutter_path = join(local_base_path, f"cluster_image_aug_2by10_cltsize{cluster_size:02}.json")
        with open(clutter_path, 'r') as f:
            clutter_lst = json.load(f)
        img_value_lst = []
        model_value_lst = []
        for clutter_data in clutter_lst:
            img_value_lst.append(int(clutter_data['isometric_path'].split('/')[1].split('_')[-1])/total_scale)
            model_value_lst.append(int(clutter_data['model_path'].split('/')[1].split('_')[-1])/total_scale)
            assert int(clutter_data['model_path'].split('/')[1].split('_')[-1]) == int(clutter_data['ptcloud_path'].split('/')[1].split('_')[-1]), "Model Ptcloud Index not Consistent"
        
        ax1.scatter([idx + 1] * len(img_value_lst),  img_value_lst, marker="o", c='goldenrod', alpha=0.8, s=markersize)
        ax1.scatter([idx + 1] * len(model_value_lst),  model_value_lst, marker="x", c='mediumblue', s=markersize + 50, linewidths=linewidth-3)
        xticks_list.append(idx + 1)
        math_idx = len(cluster_size_lst) - idx
        xticks_symbol_list.append(rf'$S_{1}I_{math_idx}$')

    plt.xticks(xticks_list, xticks_symbol_list)
    plt.ylabel("Interpolation from sphere to cube", fontsize=fontsize-10)
    ax1.tick_params(axis="x", labelsize=fontsize-10)
    ax1.tick_params(axis="y", labelsize=fontsize-10)
    ax1.legend(["Image", "Shape"],loc='upper center', bbox_to_anchor=(0.5, 1.20), fancybox=True, shadow=True, fontsize=fontsize, ncol=2)
    plt.tight_layout()
    plt.savefig(join(vis_folder, "toydataset_imageaug_distribution.png"))

def plot_imageaug_cluster_v2(local_base_path):
    cluster_size_lst = [35, 30, 25, 20, 15, 10, 5, 1]
    total_scale = 1000.0
    xticks_symbol_list = []
    xticks_list = []
    ##Plot image clutter json
    plt.style.use('ggplot')
    _, ax1 = plt.subplots(figsize=(22,18))  #22 18
    for idx, cluster_size in enumerate(cluster_size_lst):
        clutter_path = join(local_base_path, f"cluster_image_aug_2by10_cltsize{cluster_size:02}.json")
        with open(clutter_path, 'r') as f:
            clutter_lst = json.load(f)
        img_value_lst = []
        model_value_lst = []
        for clutter_data in clutter_lst:
            img_value_lst.append(int(clutter_data['isometric_path'].split('/')[1].split('_')[-1])/total_scale)
            model_value_lst.append(int(clutter_data['model_path'].split('/')[1].split('_')[-1])/total_scale)
            assert int(clutter_data['model_path'].split('/')[1].split('_')[-1]) == int(clutter_data['ptcloud_path'].split('/')[1].split('_')[-1]), "Model Ptcloud Index not Consistent"
        
        ax1.scatter([idx + 1] * len(img_value_lst),  img_value_lst, marker="o", c='goldenrod', alpha=0.8, s=markersize)
        ax1.scatter([idx + 1] * len(model_value_lst),  model_value_lst, marker="x", c='mediumblue', s=markersize + 50, linewidths=linewidth-3)
        xticks_list.append(idx + 1)
        math_idx = len(cluster_size_lst) - idx
        xticks_symbol_list.append(rf'$S_{1}I_{math_idx}$')

    plt.xticks(xticks_list, xticks_symbol_list)
    plt.yticks([])
    #plt.yticks([0, 1], ['sphere', 'cube'])
    #plt.ylabel("Interpolation", fontsize=fontsize)
    ax1.tick_params(axis="x", labelsize=fontsize+5)
    ax1.tick_params(axis="y", labelsize=fontsize-10)
    ax1.legend(["Image", "Shape"],loc='upper center', bbox_to_anchor=(0.5, 1.20), fancybox=True, shadow=True, fontsize=fontsize+5, ncol=2)
    plt.tight_layout()
    plt.savefig(join(vis_folder, "toydataset_imageaug_distribution_v2.png"), bbox_inches='tight')

def plot_shapeaug_cluster_v2(local_base_path):
    cluster_size_lst = [1, 5, 10, 15, 20, 25,30, 35]
    total_scale = 1000.0
    xticks_symbol_list = []
    xticks_list = []

    ##Plot image clutter json

    plt.style.use('ggplot')
    _, ax1 = plt.subplots(figsize=(22,18))
    for idx, cluster_size in enumerate(cluster_size_lst):
        clutter_path = join(local_base_path, f"cluster_shape_aug_2by10_cltsize{cluster_size:02}.json")
        with open(clutter_path, 'r') as f:
            clutter_lst = json.load(f)
        img_value_lst = []
        model_value_lst = []
        for clutter_data in clutter_lst:
            img_value_lst.append(int(clutter_data['isometric_path'].split('/')[1].split('_')[-1])/total_scale)
            model_value_lst.append(int(clutter_data['model_path'].split('/')[1].split('_')[-1])/total_scale)
            assert int(clutter_data['model_path'].split('/')[1].split('_')[-1]) == int(clutter_data['ptcloud_path'].split('/')[1].split('_')[-1]), "Model Ptcloud Index not Consistent"
        
        ax1.scatter([idx + 1] * len(img_value_lst),  img_value_lst, marker="o", c='goldenrod', alpha=0.8, s=markersize)
        ax1.scatter([idx + 1] * len(model_value_lst),  model_value_lst, marker="x", c='mediumblue', s=markersize + 50, linewidths=linewidth-3)
        xticks_list.append(idx + 1)
        math_idx = idx + 1
        xticks_symbol_list.append(rf'$S_{math_idx}I_{8}$')

    plt.xticks(xticks_list, xticks_symbol_list)
    plt.yticks([])
    #plt.yticks([0, 1], ['sphere', 'cube'])
    #plt.ylabel("Interpolation", fontsize=fontsize)
    #plt.ylabel("Interpolation from sphere to cube", fontsize=fontsize-10)
    ax1.tick_params(axis="x", labelsize=fontsize+5)
    ax1.tick_params(axis="y", labelsize=fontsize-10)
    ax1.legend(["Image", "Shape"],loc='upper center', bbox_to_anchor=(0.5, 1.20), fancybox=True, shadow=True, fontsize=fontsize+5, ncol=2)
    plt.tight_layout()
    plt.savefig(join(vis_folder, "toydataset_shapeaug_distribution_v2.png"), bbox_inches='tight')


def plot_shapeaug_cluster(local_base_path):
    cluster_size_lst = [1, 5, 10, 15, 20, 25,30, 35]
    total_scale = 1000.0
    xticks_symbol_list = []
    xticks_list = []

    ##Plot image clutter json

    plt.style.use('ggplot')
    _, ax1 = plt.subplots(figsize=(18,15))
    for idx, cluster_size in enumerate(cluster_size_lst):
        clutter_path = join(local_base_path, f"cluster_shape_aug_2by10_cltsize{cluster_size:02}.json")
        with open(clutter_path, 'r') as f:
            clutter_lst = json.load(f)
        img_value_lst = []
        model_value_lst = []
        for clutter_data in clutter_lst:
            img_value_lst.append(int(clutter_data['isometric_path'].split('/')[1].split('_')[-1])/total_scale)
            model_value_lst.append(int(clutter_data['model_path'].split('/')[1].split('_')[-1])/total_scale)
            assert int(clutter_data['model_path'].split('/')[1].split('_')[-1]) == int(clutter_data['ptcloud_path'].split('/')[1].split('_')[-1]), "Model Ptcloud Index not Consistent"
        
        ax1.scatter([idx + 1] * len(img_value_lst),  img_value_lst, marker="o", c='goldenrod', alpha=0.8, s=markersize)
        ax1.scatter([idx + 1] * len(model_value_lst),  model_value_lst, marker="x", c='mediumblue', s=markersize + 50, linewidths=linewidth-3)
        xticks_list.append(idx + 1)
        math_idx = idx + 1
        xticks_symbol_list.append(rf'$S_{math_idx}I_{8}$')

    plt.xticks(xticks_list, xticks_symbol_list)
    #plt.ylabel("Interpolation from sphere to cube", fontsize=fontsize-10)
    ax1.tick_params(axis="x", labelsize=fontsize-10)
    ax1.tick_params(axis="y", labelsize=fontsize-10)
    ax1.legend(["Image", "Shape"],loc='upper center', bbox_to_anchor=(0.5, 1.20), fancybox=True, shadow=True, fontsize=fontsize, ncol=2)
    plt.tight_layout()
    plt.savefig(join(vis_folder, "toydataset_shapeaug_distribution.png"))


def generate_shape_aug(local_base_path):
    ##Generate shape augmentation method 
    fixed_img_cluster_size = 35
    img_cluster_size_lst = [-1, 5, 10, 15, 20, 25, 30, 35]
    shape_cluster_size_lst = [1, 5, 10, 15, 20, 25, 30, 35]
    for idx, img_cluster_size in enumerate(img_cluster_size_lst):
        shape_cluster_size = shape_cluster_size_lst[idx]
        img_cluster_size = img_cluster_size_lst[idx]
        img_fix_clutter_path = join(local_base_path, f"cluster_image_aug_2by10_cltsize{fixed_img_cluster_size:02}.json")
        img_clutter_path = join(local_base_path, f"cluster_image_aug_2by10_cltsize{img_cluster_size:02}.json")
        shape_clutter_path = join(local_base_path, f"cluster_shape_aug_2by10_cltsize{shape_cluster_size:02}.json")
            
        with open(img_fix_clutter_path, 'r') as f:
            fix_img_clutter_lst = json.load(f)
        
        if idx == 0:
            with open(shape_clutter_path, 'w', encoding='utf-8') as f:
                json.dump(fix_img_clutter_lst, f, ensure_ascii=False, indent=4)
        else:
            with open(img_clutter_path, 'r') as f:
                img_clutter_lst = json.load(f)
            for idx in range(len(fix_img_clutter_lst)):
                fix_img_clutter_lst[idx]['model_path'] = join(*img_clutter_lst[idx]['isometric_path'].split('/')[:-1], 'model.obj')
                fix_img_clutter_lst[idx]['ptcloud_path'] = join(*img_clutter_lst[idx]['isometric_path'].split('/')[:-1], 'pt_1024.npy')
                fix_img_clutter_lst[idx]['shapekey_value'] = img_clutter_lst[idx]['shapekey_value']
                fix_img_clutter_lst[idx]['dir_path'] = img_clutter_lst[idx]['dir_path']
                
            with open(shape_clutter_path, 'w', encoding='utf-8') as f:
                json.dump(fix_img_clutter_lst, f, ensure_ascii=False, indent=4)

def check_shapeaug_overlap(local_base_path):
    cluster_size_lst = [1, 5, 10, 15, 20, 25, 30, 35]
    total_scale = 1000.0
    xticks_list = []

    ##Plot image clutter json
    for idx, cluster_size in enumerate(cluster_size_lst):
        clutter_path = join(local_base_path, f"cluster_shape_aug_2by10_cltsize{cluster_size:02}.json")
        with open(clutter_path, 'r') as f:
            clutter_lst = json.load(f)
            clutter_model_lst = [join(*clutter_lst[i]['model_path'].split('/')[:-1]) for i in range(len(clutter_lst))]
            clutter_ptcloud_lst = [join(*clutter_lst[i]['ptcloud_path'].split('/')[:-1]) for i in range(len(clutter_lst))]
            clutter_img_lst = [join(*clutter_lst[i]['isometric_path'].split('/')[:-1]) for i in range(len(clutter_lst))]
            
            no_overlap = True
            for model, ptcloud, img in zip(clutter_model_lst, clutter_ptcloud_lst, clutter_img_lst):
                if model in test_dir_lst or model in val_dir_lst:
                    no_overlap = False
                assert model == ptcloud
                if img in test_dir_lst or model in val_dir_lst:
                    no_overlap = False
        
        print(f"Cluster Size: {cluster_size}, no_overlap: {no_overlap}")

def visualize_shape_aug(local_base_path):
    cluster_size_lst = [1, 5, 10, 15, 20, 25, 30, 35]
    total_scale = 1000.0
    xticks_list = []

    ##Plot image clutter json
    for idx, cluster_size in enumerate(cluster_size_lst):
        clutter_path = join(local_base_path, f"cluster_shape_aug_2by10_cltsize{cluster_size:02}.json")
        with open(clutter_path, 'r') as f:
            clutter_lst = json.load(f)
            index_lst = list(range(10, 20))
            index_lst.reverse()
            index_lst = list(range(0, 10)) + index_lst

            for i in index_lst:
                assert join(*clutter_lst[i]['ptcloud_path'].split('/')[:-1]) == join(*clutter_lst[i]['model_path'].split('/')[:-1])
                
                tmp_pt = np.load(join(local_base_path, clutter_lst[i]['ptcloud_path']))
                tmp_img = cv2.imread(join(local_base_path, clutter_lst[i]['isometric_path']))
                tmp_pt = np.expand_dims(tmp_pt, axis=0)
                tmp_img = np.expand_dims(tmp_img, axis=0)
                if i == 0:
                    ptclouds = tmp_pt
                    images = tmp_img
                else:
                    ptclouds = np.concatenate((ptclouds, tmp_pt), axis=0)
                    images = np.concatenate((images, tmp_img), axis=0)
        
        
            print(images.shape)
            visuaize_pts(ptclouds, rows=1, cols=20, title=f'Cluster Size: {cluster_size}')
            plt.savefig(f"pts_shape_aug_cltsize{cluster_size}.png")

            image_grid(images, title=f'Cluster Size: {cluster_size}', rows=1, cols=20, colorbar=False)
            plt.savefig(f"imgs_shape_aug_cltsize{cluster_size}.png")
            plt.close()

def visualize_img_aug(local_base_path):
    cluster_size_lst = [1, 5, 10, 15, 20, 25, 30, 35]
    total_scale = 1000.0
    xticks_list = []

    ##Plot image clutter json
    for idx, cluster_size in enumerate(cluster_size_lst):
        clutter_path = join(local_base_path, f"cluster_image_aug_2by10_cltsize{cluster_size:02}.json")
        with open(clutter_path, 'r') as f:
            clutter_lst = json.load(f)
            index_lst = list(range(10, 20))
            index_lst.reverse()
            index_lst = list(range(0, 10)) + index_lst

            for i in index_lst:
                assert join(*clutter_lst[i]['ptcloud_path'].split('/')[:-1]) == join(*clutter_lst[i]['model_path'].split('/')[:-1])

                tmp_pt = np.load(join(local_base_path, clutter_lst[i]['ptcloud_path']))
                tmp_img = cv2.imread(join(local_base_path, clutter_lst[i]['isometric_path']))
                tmp_pt = np.expand_dims(tmp_pt, axis=0)
                tmp_img = np.expand_dims(tmp_img, axis=0)
                if i == 0:
                    ptclouds = tmp_pt
                    images = tmp_img
                else:
                    ptclouds = np.concatenate((ptclouds, tmp_pt), axis=0)
                    images = np.concatenate((images, tmp_img), axis=0)
            
            visuaize_pts(ptclouds, rows=1, cols=20, title=f'Cluster Size: {cluster_size}')
            plt.savefig(f" {cluster_size}.png")

            image_grid(images, title=f'Cluster Size: {cluster_size}', rows=1, cols=20, colorbar=False)
            plt.savefig(f"imgs_image_aug_cltsize{cluster_size}.png")
            plt.close()



#plot_imageaug_cluster(local_base_path)
#plot_shapeaug_cluster(local_base_path)

plot_imageaug_cluster_v2(local_base_path)
plot_shapeaug_cluster_v2(local_base_path)

#check_shapeaug_overlap(local_base_path)

# visualize_shape_aug(local_base_path)
# visualize_img_aug(local_base_path)
#plot_imageaug_cluster(local_base_path)