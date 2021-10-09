"""Dataloader Class for toydataset
"""
import torch.utils.data as data
from os.path import join
import json
import logging
import tqdm
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys
import torch
import shutil
import torchvision.transforms as transforms
sys.path.append('../../')
import dataset.pointcloud_processor as pointcloud_processor
from auxiliary.my_utils import visuaize_pts, save_image
from auxiliary.PointFlowRenderer.pointflow_fig_function import pts_to_xml

class ToyDataset(data.Dataset):
    """
    ToyDataset Dataloader
    """

    def __init__(self, data_base_dir, 
                        json_file, 
                        num_points=2500, 
                        train=True,
                        normalization="UnitBall",
                        logger=None):
        
        with open(join(data_base_dir, json_file), 'r') as f:
            data_dic_lst = json.load(f)
        
        data_dic_lst = sorted(data_dic_lst, key=lambda k:k['shapekey_value'])
        
        self.data_base_dir = data_base_dir
        self.data_dic_lst = data_dic_lst
        self.num_points = num_points
        self.normalization = normalization
        
        if logger:
            logger.info(f"Data Index Json File: {json_file}, PreLoading Ptcloud ...")
        for idx, data_dic in tqdm.tqdm(enumerate(self.data_dic_lst), total=len(self.data_dic_lst)):
            tmp_pts = np.load(join(data_base_dir, *data_dic['ptcloud_path'].split('/')[:-1], 'pt_30000.npy'))
            tmp_pts = np.expand_dims(tmp_pts, axis=0)
            if idx == 0:
                self.ptclouds = tmp_pts
            else:
                self.ptclouds = np.concatenate((self.ptclouds, tmp_pts), axis=0)

        self.init_normalization()
        self.ptclouds = torch.from_numpy(self.ptclouds)
        self.ptclouds = self.normalization_function(self.ptclouds)

        self.transforms = transforms.Compose([
                    transforms.ToTensor()
                    ])

    def init_normalization(self):
        if self.normalization == "UnitBall":
            self.normalization_function = pointcloud_processor.Normalization.normalize_unitL2ball_functional
        elif self.normalization == "BoundingBox":
            self.normalization_function = pointcloud_processor.Normalization.normalize_bounding_box_functional
        else:
            self.normalization_function = pointcloud_processor.Normalization.identity_functional

    def __len__(self):
        return len(self.data_dic_lst)

    def __getitem__(self, index):

        image = cv2.imread(join(self.data_base_dir, self.data_dic_lst[index]['isometric_path']))
        image = self.transforms(image).float()

        ptcloud = self.ptclouds[index]
        choice = np.random.choice(ptcloud.shape[0], self.num_points, replace=False)
        ptcloud = ptcloud[choice, :].contiguous().float()
        
        shapekey = self.data_dic_lst[index]['shapekey_value']

        dic = {"image": image, "points": ptcloud, "shapekey": shapekey}

        return dic
         


    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    
    #json_file = 'val_interp_1000.json'
    #json_file = 'cluster_shape_aug_2by10_cltsize35.json'
    cltsize_lst = [1, 35]  # 5, 10, 15, 20, 25, 30
    folder = "vis/augment_demo_submission"
    for cltsize in cltsize_lst:
        #json_file = f'cluster_shape_aug_2by10_cltsize{cltsize:02}.json'
        json_file = f'cluster_image_aug_2by10_cltsize{cltsize:02}.json'
        toydata = ToyDataset(data_base_dir='../../../data/cubesphere_1000/', 
                json_file=json_file,
                num_points=2500,
                train=True, logger=logger)

        dataloader = torch.utils.data.DataLoader(toydata, batch_size=20,
                                            shuffle=False,
                                            num_workers=4)
        for batch in dataloader:
            print(batch['shapekey'])

        # for batch in dataloader:
        #     for idx, points in enumerate(batch['points']):
        #         #points = points.unsqueeze(0)
        #         points = points.numpy()
        #         pts_to_xml(points, join(folder, 'points', f'points_idx{idx}_'+json_file.replace('json', 'xml')))
            #visuaize_pts(batch['points'], title=f'2by10_cltsize{cltsize:02}', rows=3, cols=8)
            #plt.savefig(join('vis/dataloader', 'points_'+json_file.replace('json', 'png')))
            #save_image(batch['image'], join('vis/dataloader', 'image_'+json_file.replace('json', 'png')))
        
        
            