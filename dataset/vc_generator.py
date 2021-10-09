"""This script is used to generate viewer-center data in SVR tasks on ShapeNet 13 
    For a review of viewer-center, see 
    Pixels, voxels, and views: A study of shape representations for single view 3D object shape prediction
    https://arxiv.org/abs/1804.06032
"""

"""
Initialize a dataset object, sample some indexes of data pair, save it to local
"""
import re
import os
import sys
import random
sys.path.append('..')
#from auxiliary.argument_parser import parser
from auxiliary.my_utils import plant_seeds
from dataset.dataset_shapenet import ShapeNet
from shutil import copytree, copy
from pathlib import Path
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2

# SAMPLE_SIZE=30
# DOWN_SAMPLES=3000
# sample_folder = "data/vc_samples"
# opt = parser()
# plant_seeds(opt.manual_seed)
# opt.shapenet13 = True

def unit(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def camera_info(param):
    theta = np.deg2rad(param[0])
    phi = np.deg2rad(param[1])

    camY = param[3]*np.sin(phi)
    temp = param[3]*np.cos(phi)
    camX = temp * np.cos(theta)    
    camZ = temp * np.sin(theta)        
    cam_pos = np.array([camX, camY, camZ])        

    axisZ = cam_pos.copy()
    axisY = np.array([0,1,0])
    axisX = np.cross(axisY, axisZ)
    axisY = np.cross(axisZ, axisX)

    cam_mat = np.array([unit(axisX), unit(axisY), unit(axisZ)])
    return cam_mat, cam_pos



def sample_from_shapenet():
    """Get a random sample from ShapeNet 13 train set to implement VC generation in local 
    """
    sample_path_list = []
    shapenet = ShapeNet(opt, train=True)
    sample_indexes = random.sample(range(len(shapenet)), SAMPLE_SIZE)

    for idx in tqdm.tqdm(sample_indexes, total=len(sample_indexes)):
        sample = shapenet[idx]
        origin_pt_path = sample['pointcloud_path']
        origin_img_folder = sample['image_path']

        img_idx = [i.start() for i in re.finditer('/', origin_img_folder)][6]
        new_img_folder = os.path.join(sample_folder, origin_img_folder[img_idx + 1:])
        if not os.path.exists(new_img_folder):
            copytree(origin_img_folder, new_img_folder)

        pt_idx = [i.start() for i in re.finditer('/', origin_pt_path)][6]
        origin_pt_folder = os.path.split(origin_pt_path)[0]
        new_pt_path = os.path.join(sample_folder, origin_pt_path[pt_idx + 1:])
        new_pt_folder = os.path.split(new_pt_path)[0]

        if not os.path.exists(new_pt_folder):
            Path(new_pt_folder).mkdir(parents=True, exist_ok=True)

        copy(origin_pt_path, new_pt_path)
        sample_path_list.append((new_img_folder, new_pt_path))

    with open(os.path.join(sample_folder, "sample_paths.json"), "w") as outfile: 
        json.dump(sample_path_list, outfile)


def vc_generator():
    with open(os.path.join(sample_folder, "sample_paths.json"), 'r') as openfile: 
    # Reading from json file 
        json_object = json.load(openfile)
    for sample in json_object:
        img_folder, pt_path = sample[0], sample[1]
        view_path = os.path.join(img_folder, 'rendering_metadata.txt')
        cam_params = np.loadtxt(view_path)
        pts = np.load(pt_path)
        pt_idx = random.sample(range(pts.shape[0]), DOWN_SAMPLES)
        pts = pts[pt_idx]
        #plot_ptcloud(pts)
        for index, param in enumerate(cam_params):
            cam_mat, _ = camera_info(param)
            pt_trans = np.dot(pts, cam_mat.transpose())
            img_path = os.path.join(img_folder, '%02d.png'%index)
            img = cv2.imread(img_path)
            #plot(pt_trans, img)
            #theta, phi = param[0], param[1]
            #print(theta, phi)
            '''
            print(img_path)
            '''
            #break
        break


def plot(pts, img, title="None", ifsave=True, path='', scale=0.5):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(pts[:,2], pts[:,0], pts[:,1], s=2)
    #ax.scatter(pts[:,0], pts[:,2], pts[:,1], s=2)
    ax.set_xlim([-1*scale, scale])
    ax.set_ylim([-1*scale, scale])
    ax.set_zlim([-1*scale, scale])
    ax.set_title("GT view (0,0)")
    ax.view_init(0, 0)

    ax = fig.add_subplot(122)
    ax.imshow(img)
    ax.set_title(title)
    if ifsave:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()

def plot_pred(gt_pts, pred_pts, img, title="None", ifsave=True, path='', scale=0.5):
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(131)
    ax.imshow(img)
    ax.set_title(title)

    ax = fig.add_subplot(132, projection='3d')
    ax.scatter(gt_pts[:,2], gt_pts[:,0], gt_pts[:,1], s=2)
    #ax.scatter(pts[:,0], pts[:,2], pts[:,1], s=2)
    ax.set_xlim([-1*scale, scale])
    ax.set_ylim([-1*scale, scale])
    ax.set_zlim([-1*scale, scale])
    ax.set_title("GT view (0,0)")
    ax.view_init(0, 0)


    ax = fig.add_subplot(133, projection='3d')
    ax.scatter(pred_pts[:,2], pred_pts[:,0], pred_pts[:,1], s=2)
    #ax.scatter(pts[:,0], pts[:,2], pts[:,1], s=2)
    ax.set_xlim([-1*scale, scale])
    ax.set_ylim([-1*scale, scale])
    ax.set_zlim([-1*scale, scale])
    ax.set_title("PRED view (0,0)")
    ax.view_init(0, 0)
    plt.tight_layout()

    if ifsave:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    vc_generator()





