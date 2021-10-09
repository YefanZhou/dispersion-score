import torch
import torch.nn as nn
#import auxiliary.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D
from auxiliary.ChamferDistancePytorch.fscore import fscore
import os
import training.metro as metro
from joblib import Parallel, delayed
import numpy as np
from dataset.vc_generator import plot

def chamfer_distance(a,b):
    """pytorch version chamfer distance
    """
    x,y = a,b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2,1))
    yy = torch.bmm(y, y.transpose(2,1))
    zz = torch.bmm(x, y.transpose(2,1))
    diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2,1) + ry - 2*zz)
    
    return torch.min(P, 1)[0], torch.min(P, 2)[0], torch.min(P, 1)[1], torch.min(P, 2)[1]


class TrainerLoss(object):
    """
    This class implements all functions related to the loss of Atlasnet, mainly applies chamfer and metro.
    """

    def __init__(self):
        super(TrainerLoss, self).__init__()

    def build_losses(self):
        """
        Create loss functions.
        """
        if not self.opt.no_compile_chamfer:
            import auxiliary.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D
            self.distChamfer = dist_chamfer_3D.chamfer_3DDist()
        else:
            self.distChamfer = chamfer_distance

        self.loss_model = self.chamfer_loss

    def fuse_primitives(self):
        """
        Merge generated surface elements in a single one and prepare data for Chamfer
        Input size : batch, prim, 3, npoints
        Output size : prim, prim*npoints, 3
        :return:
        """
        #
        if self.opt.network == 'atlasnet':
            self.data.pointsReconstructed = self.data.pointsReconstructed_prims.transpose(2, 3).contiguous()
            self.data.pointsReconstructed = self.data.pointsReconstructed.view(self.batch_size, -1, 3)
        elif self.opt.network == 'psgn' or self.opt.network == 'foldnet':
            self.data.pointsReconstructed = self.data.pointsReconstructed_prims
        else:
            raise NotImplementedError(f"{self.opt.network} is not implemented")
        

    def chamfer_loss(self):
        """
        Training loss of Atlasnet. The Chamfer Distance. Compute the f-score in eval mode.
        :return:
        """
        inCham1 = self.data.points.view(self.data.points.size(0), -1, 3).contiguous()
        inCham2 = self.data.pointsReconstructed.contiguous().view(self.data.points.size(0), -1, 3).contiguous()
        dist1, dist2, idx1, idx2 = self.distChamfer(inCham1, inCham2)  # mean over points
        self.data.loss = torch.mean(dist1) + torch.mean(dist2)  # mean over points
        if not self.flags.train:
            self.data.loss_fscore, _, _ = fscore(dist1, dist2)
            self.data.loss_fscore = self.data.loss_fscore.mean()
        
    def metro(self):
        """
        Compute the metro distance on a randomly selected test files.
        Uses joblib to leverage as much cpu as possible
        :return:
        """
        metro_path = './dataset/data/metro_files'
        metro_files_path = '/'.join([metro_path, 'files-metro.txt'])
        self.metro_args_input = []
        if not os.path.exists(metro_files_path):
            os.system("chmod +x dataset/download_metro_files.sh")
            os.system("./dataset/download_metro_files.sh")
        ext = '.png' if self.opt.SVR else '.npy'
        with open(metro_files_path, 'r') as file:
            files = file.read().split('\n')

        for file in files:
            if file[-3:] == "ply":
                cat = file.split('/')[0]
                name = file.split('/')[1][:-4]
                input_path = '/'.join([metro_path, cat, name + ext])
                input_path_points = '/'.join([metro_path, cat, name + '.npy'])
                gt_path = '/'.join([metro_path, cat, name + '.ply'])
                path = self.demo(input_path, input_path_points)
                self.metro_args_input.append((path, gt_path))

        print("start metro calculus. This is going to take some time (30 minutes)")
        self.metro_results = Parallel(n_jobs=-1, backend="multiprocessing")(
            delayed(metro.metro)(*i) for i in self.metro_args_input)
        self.metro_results = np.array(self.metro_results).mean()
        print(f"Metro distance : {self.metro_results}")



