import os
import argparse
import tqdm
import trimesh
import numpy as np
from os.path import join
import glob
import open3d as o3d
import logging

class DisableLogger():
    def __enter__(self):
       logging.disable(logging.CRITICAL)
    def __exit__(self, a, b, c):
       logging.disable(logging.NOTSET)

num_points = 30000
base_data_path = '../../../data/cubesphere_1000/CubeSphere_interp_1000'
data_dirs = glob.glob(join(base_data_path, '*'))
data_dirs.sort()

print(len(data_dirs))
for data_dir in tqdm.tqdm(data_dirs, total=len(data_dirs)):
    mesh_path = join(data_dir, 'model.obj')
    pt_save_path = join(data_dir, f'pt_{num_points}.npy')

    with DisableLogger():
        mesh_or_scene = trimesh.load_mesh(mesh_path)

    obj_mesh = mesh_or_scene
    #obj_mesh = trimesh_remove_texture(mesh_or_scene)

    obj_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(obj_mesh.vertices), 
                                        o3d.utility.Vector3iVector(obj_mesh.faces))
    ptcloud = obj_mesh.sample_points_uniformly(number_of_points = num_points)
    ptcloud_np = np.asarray(ptcloud.points)
    np.save(pt_save_path, ptcloud_np)
    print(f" Save Npy to {pt_save_path} ")
    