
from shutil import copytree
from matplotlib import pyplot as plt
import torch.utils.data as data
import os.path
import torch
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
import sys
sys.path.append("../")
from dataset.image_aug.AutoAugment.autoaugment import ImageNetPolicy, CIFAR10Policy, SVHNPolicy
from dataset.image_aug.AutoAugment.autoaugment_param import PolicySeq, ColorAugFore, GeometricPolicySeq, PolicySeqBlackBG, GeometricPolicyMag
import auxiliary.my_utils as my_utils
from auxiliary.constant import TRAIN_SIZE, TEST_SIZE, MAXVIEW_PER_SHAPE
import pickle
from os.path import join, dirname, exists
from easydict import EasyDict
import json
import math
import logging
from auxiliary.my_utils import image_grid
from dataset.vc_generator import camera_info, plot
import dataset.pointcloud_processor as pointcloud_processor
from copy import deepcopy
import tqdm
import random
from distutils.dir_util import copy_tree
from pathlib import Path


class ShapeNet(data.Dataset):
    """
    Shapenet Dataloader
    Uses Shapenet V1
    """

    def __init__(self, opt, train=True, num_image_per_object=1):
        self.opt = opt
        if opt.no_compile_chamfer:
            self.num_sample = opt.number_points if train else opt.number_points_eval
        else:
            self.num_sample = opt.number_points if train else 2500
            
        self.train = train
        self.init_normalization()
        self.init_singleview()
        assert num_image_per_object <= 24, 'ShapeNet 13 R2N2 rendering only have 24 views per shape'

        if not opt.demo:
            self.opt.logger.info('Create Shapenet Dataset Train Set...') if train else self.opt.logger.info('Create Shapenet Dataset Test Set...')
            # Define core path array
            self.datapath = []
            self.category_datapath = {}

            # Load classes
            print(f"Rendering root dir {opt.rendering_root_dir}")
            self.pointcloud_path = join(dirname(__file__), 'data/ShapeNetV1PointCloud')
            self.image_path = opt.rendering_root_dir

            # Load taxonomy file
            self.taxonomy_path = join(dirname(__file__), 'data/taxonomy.json')
            if not exists(self.taxonomy_path):
                os.system("chmod +x dataset/download_shapenet_pointclouds.sh")
                os.system("./dataset/download_shapenet_pointclouds.sh")

            self.classes = [x for x in next(os.walk(self.pointcloud_path))[1]]
            with open(self.taxonomy_path, 'r') as f:
                self.taxonomy = json.load(f)

            self.id2names = {}
            self.names2id = {}
            for dict_class in self.taxonomy:
                if dict_class['synsetId'] in self.classes:
                    name = dict_class['name'].split(sep=',')[0]
                    self.id2names[dict_class['synsetId']] = name
                    self.names2id[name] = dict_class['synsetId']

            # Select classes
            if opt.shapenet13:
                opt.class_choice = ["airplane", "bench", "cabinet", "car", "chair", "display", "lamp", "loudspeaker",
                                    "rifle", "sofa", "table", "telephone", "vessel"]

            if len(opt.class_choice) > 0:
                new_classes = []
                for category in opt.class_choice:
                    new_classes.append(self.names2id[category])
                self.classes = new_classes

            # Create Cache path
            self.path_dataset = join(dirname(__file__), 'data', 'cache')
            if not exists(self.path_dataset):
                os.mkdir(self.path_dataset)
            self.path_dataset = join(self.path_dataset,
                                     self.opt.normalization + str(train) + "_".join(self.opt.class_choice))

            if not exists(self.image_path):
                os.system("chmod +x dataset/download_shapenet_renderings.sh")
                os.system("./dataset/download_shapenet_renderings.sh")

            self.num_image_per_object = num_image_per_object
            self.idx_image_val = 0
            
            #--------------------------------------------------------------------------#
            # Compile list of pointcloud path by selected category
            for category in self.classes:
                dir_pointcloud = join(self.pointcloud_path, category)
                dir_image = join(self.image_path, category)
                list_pointcloud = sorted(os.listdir(dir_pointcloud))

                if self.train:
                    list_pointcloud = list_pointcloud[:int(len(list_pointcloud) * 0.8)]
                else:
                    list_pointcloud = list_pointcloud[int(len(list_pointcloud) * 0.8):]

                self.opt.logger.info(
                    '    category '
                    + category
                    + "  "
                    + self.id2names[category]
                    + ' Number Files :'
                    + str(len(list_pointcloud))
                )

                if len(list_pointcloud) != 0:
                    self.category_datapath[category] = []
                    for pointcloud in list_pointcloud:
                        pointcloud_path = join(dir_pointcloud, pointcloud)       #data/ShapeNetV1PointCloud/04530566/ffffe224db39febe288b05b36358465d.points.ply.npy
                        image_folder = join(dir_image, pointcloud.split(".")[0], "easy")
                        view_path = os.path.join(image_folder, 'rendering_metadata.txt')
                        cam_params = np.loadtxt(view_path)
                        cam_rotmat, _ = camera_info(cam_params)
                        image_path = join(image_folder, ShapeNet.int2str(self.idx_image_val) + ".png")
                        if not self.opt.SVR or exists(image_path):
                           self.category_datapath[category].append((pointcloud_path, image_path, pointcloud, category, cam_rotmat))
                        else:
                           self.opt.logger.info(f"Rendering not found : {image_path}")

            # Add all retained path to a global vector
            for item in self.classes:
                for pointcloud in self.category_datapath[item]:
                    self.datapath.append(pointcloud)
            #------------------------------------------------------------------------------#

            # Preprocess and cache files
            self.preprocess()
            #self.preprocess_wo_cached()

    def preprocess(self):
        if exists(self.path_dataset + "info.pkl"):
            # Reload dataset
            self.opt.logger.info(f"Reload dataset : {self.path_dataset}")
            with open(self.path_dataset + "info.pkl", "rb") as fp:
                self.data_metadata = pickle.load(fp)

            self.data_points = torch.load(self.path_dataset + "points.pth")
        else:
            assert self.num_image_per_object == 1, "preprocessing must be conducted when num_image_per_object = 1"
            # Preprocess dataset and put in cache for future fast reload
            self.opt.logger.info("preprocess dataset...")
            self.datas = [self._getitem(i) for i in range(self.__len__())]

            # Concatenate all proccessed files
            self.data_points = [a[0] for a in self.datas]
            self.data_points = torch.cat(self.data_points, 0)

            self.data_metadata = [{'pointcloud_path': a[1], 'image_path': a[2], 'name': a[3], 'category': a[4], 'cam_rotmat': a[5]} for a in
                                  self.datas]

            # Save in cache
            with open(self.path_dataset + "info.pkl", "wb") as fp:  # Pickling
                pickle.dump(self.data_metadata, fp)
            torch.save(self.data_points, self.path_dataset + "points.pth")

        #self.opt.logger.info("Dataset Size: " + str(len(self.data_metadata)))
        self.opt.logger.info(f"Dataset Shape Size: {self.data_points.shape[0]} Sample Size: {len(self.datapath)}")
        self.opt.logger.info("###############################################################")

    def init_normalization(self):
        if not self.opt.demo:
            self.opt.logger.info("Dataset normalization : " + self.opt.normalization)

        if self.opt.normalization == "UnitBall":
            self.normalization_function = pointcloud_processor.Normalization.normalize_unitL2ball_functional
        elif self.opt.normalization == "BoundingBox":
            self.normalization_function = pointcloud_processor.Normalization.normalize_bounding_box_functional
        else:
            self.normalization_function = pointcloud_processor.Normalization.identity_functional

    def init_singleview(self):
        ## Define Image Transforms
        if self.opt.img_aug:
            self.opt.logger.info("SVR TASK: random img crop applied in Training!")
        else:
            self.opt.logger.info("SVR TASK: NO img crop applied")

        # Transform Resize
#############################################################################################
#############################################################################################

        self.validating = transforms.Compose([
            transforms.ToTensor()
        ])

        # RandomResizedCrop or RandomCrop

        print(f"Test Augment: {self.opt.test_augment}")

    def binarize(self, img):
        """binarize the input image, background is transparent(alpha=0)
        """
        img = self.center_crop(img)
        img = self.resize(img)
        img = np.array(img)
        img = img[:, :, 3] > 0
        img = img.astype(float)
        img = transforms.ToTensor()(img)
        img = img.repeat(3, 1, 1)

        return img

    def _getitem(self, index):
        pointcloud_path, image_path, pointcloud, category, cam_rotmat = self.datapath[index]
        points = np.load(pointcloud_path)
        points = torch.from_numpy(points).float()
        points[:, :3] = self.normalization_function(points[:, :3])
        return points.unsqueeze(0), pointcloud_path, image_path, pointcloud, category, cam_rotmat

    #Origin AtlasNet version 
    def __getitem__(self, index):
        #return_dict = deepcopy(self.data_metadata[index])
        sample_dic = self.datapath[index]
        return_dict = { 'pointcloud_path': sample_dic[0], 
                        'image_path': sample_dic[1], 
                        'name': sample_dic[2], 
                        'category': sample_dic[3],
                        'cam_rotmat': torch.from_numpy(sample_dic[4]).float()}
        # Point processing
        points = self.data_points[math.floor(index / self.num_image_per_object)]
        points = points.clone()
        if self.opt.sample:
            choice = np.random.choice(points.size(0), self.num_sample, replace=True)
            points = points[choice, :]
        return_dict['points'] = points[:, :3].contiguous()

        # Image processing
        if self.opt.SVR:
            im = Image.open(return_dict['image_path'])
            im = self.validating(im)                    # to tensor      
            im = im.float()
            im = im[:3, :, :]
            return_dict['image'] = im

        return return_dict


    def __len__(self):
        return len(self.datapath)

    @staticmethod
    def int2str(N):
        if N < 10:
            return "0" + str(N)
        else:
            return str(N)

    def load(self, path):
        ext = path.split('.')[-1]
        if ext == "npy" or ext == "ply" or ext == "obj":
            return self.load_point_input(path)
        else:
            return self.load_image(path)

    def load_point_input(self, path):
        ext = path.split('.')[-1]
        if ext == "npy":
            points = np.load(path)
        elif ext == "ply" or ext == "obj":
            import pymesh
            points = pymesh.load_mesh(path).vertices
        else:
            self.opt.logger.info("invalid file extension")

        points = torch.from_numpy(points).float()
        operation = pointcloud_processor.Normalization(points, keep_track=True)
        if self.opt.normalization == "UnitBall":
            operation.normalize_unitL2ball()
        elif self.opt.normalization == "BoundingBox":
            operation.normalize_bounding_box()
        else:
            pass
        return_dict = {
            'points': points,
            'operation': operation,
            'path': path,
        }
        return return_dict


    def load_image(self, path):
        im = Image.open(path)
        im = self.validating(im)
        im = self.transforms(im)
        im = im[:3, :, :]
        return_dict = {
            'image': im.unsqueeze_(0),
            'operation': None,
            'path': path,
        }
        return return_dict

    


if __name__ == '__main__':
    print('Testing Shapenet dataset')

    # opt = {"normalization": "UnitBall", "class_choice": ["plane"], "SVR": True, "sample": True, "npoints": 2500,
    #        "shapenet13": True, "demo": False, "logger": logging.getLogger(), "img_aug": False, 
    #        "img_aug_type": 'coloraugment', "color_aug_factor":[0.3, 1, 1, 0],  "manual_seed": 1, "no_compile_chamfer": False,
    #        "autoaug_type": 'None', "number_points":2500, "mag_idx":4, "prob":0.5, "n_op":0, "magnitude":30, 
    #        "test_augment": False}


    # opt = {"normalization": "UnitBall", "class_choice": ["plane"], "SVR": True, "sample": True, "npoints": 2500,
    #        "shapenet13": True, "demo": False, "logger": logging.getLogger(), "img_aug": False, 
    #        "img_aug_type": 'geometricaugment', "color_aug_factor":[0.3, 1, 1, 0],  "manual_seed": 1, "no_compile_chamfer": False,
    #        "autoaug_type": 'None', "number_points":2500, "mag_idx":4, "prob":0.5, "n_op":1, "magnitude":30,
    #        "test_augment": False}

    # opt = {"normalization": "UnitBall", "class_choice": ["plane"], "SVR": True, "sample": True, "npoints": 2500,
    #        "shapenet13": True, "demo": False, "logger": logging.getLogger(), "img_aug": False, 
    #        "img_aug_type": 'autoaugment', "color_aug_factor":[0.3, 1, 1, 0],  "manual_seed": 1, "no_compile_chamfer": False,
    #        "autoaug_type": 'SeqBlackBG', "number_points":2500, "mag_idx":4, "prob":0.5, "n_op":2, "magnitude":30,
    #        "test_augment": False}

    # for mag_idx in [0, 1, 2, 3]:
    #     opt = {"normalization": "UnitBall", "class_choice": ["plane"], "SVR": True, "sample": True, "npoints": 2500,
    #         "shapenet13": True, "demo": False, "logger": logging.getLogger(), "img_aug": False, 
    #         "img_aug_type": 'geometricaugmentMag', "color_aug_factor":[0.3, 1, 1, 0],  "manual_seed": 1, "no_compile_chamfer": False,
    #         "autoaug_type": 'None', "number_points":2500, "mag_idx":mag_idx, "prob":0.5, "n_op":2, "magnitude":0,
    #         "test_augment": False}


  
