from __future__ import division
import os
import sys
import torch
import tqdm
import time
import math
import numpy as np
from scipy.stats import gamma
import torch.nn as nn
from os.path import join
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids 
import sklearn.metrics as sk_metrics
sys.path.append(join(os.path.dirname(os.path.abspath(__file__)), "../"))
from eval.vgg import Vgg16
from auxiliary.my_utils import chunks
if torch.cuda.is_available():
    import auxiliary.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    if y is None:
        dist = dist - torch.diag(dist.diag())
    
    dist = dist/x.shape[-1]

    return dist

def compute_ptcloud_dismatrix(X1, X2, distance_metric, logger=None):

    """return a complete distance matrix between ptcloud X1 ptcloud X2
    Params:
    ----------------------------------
    X1: (N, ptnum, 3) torch.tensor
        point cloud set 1
    X2: (N, ptnum, 3) torch.tensor
        point cloud set 2
    distance_metric: func
        metric to measure the distance of two point cloud
    Returns:
    -------------------------#---------
    D: (ptnum, ptnum) torch.tensor
        distance matrix
    """
    
    N = X1.shape[0]
    # initialize distance matrix
    D = torch.zeros([N, N])
    # iterate over one group of ptcloud
    num_compt = N * N / 2
    pbar = tqdm.tqdm(total=num_compt, desc=f"computing....DM")
    counter = 0
    report = False
    start_time = time.time()
    with torch.set_grad_enabled(False):
        for i in range(0, N):
            for j in range(i + 1, N):
                D[i, j] = distance_metric(X1[i].unsqueeze(0), X2[j].unsqueeze(0))
                D[j, i] = D[i, j]
                pbar.update(1)
                counter += 1 
                if (counter % 20000 == 0 and counter != 0 and not report):
                    elap_min = (time.time() - start_time) / 60 
                    remain_min = (elap_min / counter) * (num_compt - counter)
                    logger.info(f"point cloud Remaining time {remain_min:2f} min")
                    report = True
        pbar.close()
        
    return D

def compute_ptcloud_dismatrix_batch(X1, X2, distance_metric, batch_size, device, logger=None):

    """return a complete distance matrix between ptcloud X1 ptcloud X2
    Params:
    ----------------------------------
        X1: (N, ptnum, 3) torch.tensor
            point cloud set 1
        X2: (M, ptnum, 3) torch.tensor
            point cloud set 2
        distance_metric: func
            metric to measure the distance of two point cloud
    Returns:
    ----------------------------------
        D: (ptnum, ptnum) torch.tensor
            distance matrix
    """
    
    N = X1.shape[0]
    M = X2.shape[0]
    assert N == M,  "X1 X2 have different shape"
    # initialize distance matrix
    D = torch.zeros([N, N])
    D = D.to(device)
    # iterate over one group of ptcloud
    num_compt = int(N * N / (2 * batch_size))
    pbar = tqdm.tqdm(total=num_compt, desc=f"computing....DM")
    counter = 0
    report = False
    start_time = time.time()
    with torch.set_grad_enabled(False):
        for i in range(0, N):
            for batch_indexes in chunks(list(range(i + 1, N)), batch_size):
                x2 = X2[batch_indexes[0] : batch_indexes[0] + len(batch_indexes)]
                x1 = X1[[i]]
                x1 = x1.expand(x2.shape[0], -1, -1)
                losses = distance_metric(x1, x2, 'list')
                D[i, batch_indexes[0] : batch_indexes[0] + len(batch_indexes)] = losses
                D[batch_indexes[0] : batch_indexes[0] + len(batch_indexes), i] = losses
                pbar.update(1)
                counter += 1 
                if (counter % 20000 == 0 and counter != 0 and not report):
                    elap_min = (time.time() - start_time) / 60 
                    remain_min = (elap_min / counter) * (num_compt - counter)
                    logger.info(f"point cloud Remaining time {remain_min:2f} min")
                    report = True
        pbar.close()
        
    return D

def compute_ptcloud_dismatrix_batch(X1, X2, distance_metric, batch_size, device, logger=None):

    """return a complete distance matrix between ptcloud X1 ptcloud X2
    Params:
    ----------------------------------
        X1: (N, ptnum, 3) torch.tensor
            point cloud set 1
        X2: (M, ptnum, 3) torch.tensor
            point cloud set 2
        distance_metric: func
            metric to measure the distance of two point cloud
    Returns:
    ----------------------------------
        D: (ptnum, ptnum) torch.tensor
            distance matrix
    """
    
    N = X1.shape[0]
    M = X2.shape[0]
    assert N == M,  "X1 X2 have different shape"
    # initialize distance matrix
    D = torch.zeros([N, N])
    D = D.to(device)
    # iterate over one group of ptcloud
    num_compt = int(N * N / (2 * batch_size))
    pbar = tqdm.tqdm(total=num_compt, desc=f"computing....DM")
    counter = 0
    report = False
    start_time = time.time()
    with torch.set_grad_enabled(False):
        for i in range(0, N):
            for batch_indexes in chunks(list(range(i + 1, N)), batch_size):
                x2 = X2[batch_indexes[0] : batch_indexes[0] + len(batch_indexes)]
                x1 = X1[[i]]
                x1 = x1.expand(x2.shape[0], -1, -1)
                losses = distance_metric(x1, x2, 'list')
                D[i, batch_indexes[0] : batch_indexes[0] + len(batch_indexes)] = losses
                D[batch_indexes[0] : batch_indexes[0] + len(batch_indexes), i] = losses
                pbar.update(1)
                counter += 1 
                if (counter % 20000 == 0 and counter != 0 and not report):
                    elap_min = (time.time() - start_time) / 60 
                    remain_min = (elap_min / counter) * (num_compt - counter)
                    logger.info(f"point cloud Remaining time {remain_min:2f} min")
                    report = True
        pbar.close()
        
    return D

def compute_img_dismatrix_batch(X1, X2, distance_metric, batch_size, device, logger=None):

    """return a complete distance matrix between ptcloud X1 ptcloud X2
    Params:
    ----------------------------------
        X1: (N, ptnum, 3) torch.tensor
            point cloud set 1
        X2: (M, ptnum, 3) torch.tensor
            point cloud set 2
        distance_metric: func
            metric to measure the distance of two point cloud
    Returns:
    ----------------------------------
        D: (ptnum, ptnum) torch.tensor
            distance matrix
    """
    
    N = X1.shape[0]
    M = X2.shape[0]
    assert N == M,  "X1 X2 have different shape"
    # initialize distance matrix
    D = torch.zeros([N, N])
    D = D.to(device)
    # iterate over one group of ptcloud
    num_compt = int(N * N / (2 * batch_size))
    pbar = tqdm.tqdm(total=num_compt, desc=f"computing....DM")
    counter = 0
    report = False
    start_time = time.time()
    with torch.set_grad_enabled(False):
        for i in range(0, N):
            for batch_indexes in chunks(list(range(i + 1, N)), batch_size):
                x2 = X2[batch_indexes[0] : batch_indexes[0] + len(batch_indexes)]
                x1 = X1[[i]]

                x1 = x1.expand(x2.shape[0], -1)
                losses = torch.mean(distance_metric(x1, x2), dim=1)
                D[i, batch_indexes[0] : batch_indexes[0] + len(batch_indexes)] = losses
                D[batch_indexes[0] : batch_indexes[0] + len(batch_indexes), i] = losses
                pbar.update(1)
                counter += 1 
                if (counter % 20000 == 0 and counter != 0 and not report):
                    elap_min = (time.time() - start_time) / 60 
                    remain_min = (elap_min / counter) * (num_compt - counter)
                    logger.info(f"Image Remaining time {remain_min:2f} min")
                    report = True
        pbar.close()
        
    return D


def compute_diff_ptcloud_dismatrix_batch(X1, X2, distance_metric, batch_size, device, logger=None):

    """return a complete distance matrix between ptcloud X1 ptcloud X2
    Params:
    ----------------------------------
        X1: (N, ptnum, 3) torch.tensor
            point cloud set 1
        X2: (M, ptnum, 3) torch.tensor
            point cloud set 2
        distance_metric: func
            metric to measure the distance of two point cloud
    Returns:
    ----------------------------------
        D: (ptnum, ptnum) torch.tensor
            distance matrix
    """
    
    N = X1.shape[0]
    M = X2.shape[0]
    # initialize distance matrix
    D = torch.zeros([N, M])
    D = D.to(device)
    # iterate over one group of ptcloud
    num_compt = int(N * M / batch_size)
    pbar = tqdm.tqdm(total=num_compt, desc=f"computing....DM")
    counter = 0
    report = False
    start_time = time.time()
    with torch.set_grad_enabled(False):
        for i in range(0, N):
            for batch_indexes in chunks(list(range(0, M)), batch_size):
                x2 = X2[batch_indexes[0] : batch_indexes[0] + len(batch_indexes)]
                x1 = X1[[i]]
                x1 = x1.expand(x2.shape[0], -1, -1)
                losses = distance_metric(x1, x2, 'list')
                D[i, batch_indexes[0] : batch_indexes[0] + len(batch_indexes)] = losses
                pbar.update(1)
                counter += 1 
                if (counter % 20000 == 0 and counter != 0 and not report):
                    elap_min = (time.time() - start_time) / 60 
                    remain_min = (elap_min / counter) * (num_compt - counter)
                    logger.info(f"point cloud Remaining time {remain_min:2f} min")
                    report = True
        pbar.close()

    return D

def get_partition(matrix, preference, damping=0.75, random_state=1):
    cl = AffinityPropagation(damping=damping, affinity='precomputed', preference=preference, random_state=random_state)
    cl.fit(matrix)
    partition = cl.labels_
    return partition


def cal_pref(mat, pc=10):			
    return np.percentile(mat, pc)  

def transform_mat(mat):
    mat = -np.exp(mat/np.amax(mat))
    np.fill_diagonal(mat, 0)
    return mat

def transform_mat_std(mat, aff_diag=-1):
    mat = -np.exp(mat/abs(np.std(mat)))
    np.fill_diagonal(mat, aff_diag)
    return mat

def cluster_eval(c_method, e_method, distance_matrix, seed=0, n_cluster=10, pc=50, mean=False, aff_diag=-1):
    """calculate clustering score based on specific clustering method and evaluation score

    Params:
    ---------------------

    c_method: string
        choices: ['KMedoids', 'AP', 'OPTICS']

    e_method: string
        choices:['Inertia']

    distance_matrix: numpy.array

    """
    assert c_method in ['KMedoids', 'AP', 'OPTICS'], f"Clustering Method {c_method} not implemented"
    assert e_method in ['Inertia'], f"Evaluation Method {e_method} not implemented"
    
    if c_method == 'AP':
        # Affinity Propagation
        distance_matrix_tr = transform_mat_std(distance_matrix, aff_diag=aff_diag)     # -e^(x/std(x)) then fill the diagonal with aff_diag
        part_preference = cal_pref(distance_matrix_tr, pc=pc)                          # in increasing order, float number in last pc% position in this matrix
        matrix_part = get_partition(distance_matrix_tr, preference=part_preference, random_state=seed)

    elif c_method == 'KMedoids':
        # KMedoids
        kmedoids = KMedoids(n_clusters=n_cluster, 
                    random_state=seed, metric='precomputed', init='k-medoids++').fit(distance_matrix)
        matrix_part = kmedoids.labels_

    elif c_method == 'Hierarchical':
        clustering = AgglomerativeClustering(affinity='precomputed', n_clusters=n_cluster, linkage='complete').fit(distance_matrix) # average complete
        matrix_part = clustering.labels_

    if e_method == 'Inertia':
        if c_method != 'KMedoids':
            score = inertia(distance_matrix, matrix_part)
        else:
            score = kmedoids.inertia_
    
    if mean:
        score = score / len(matrix_part)

    return score, matrix_part
        

def inertia(distance_matrix, matrix_part):
    """Return inertia of clustering partition (sum of intra-cluster distance)

    Params:
    --------------------
    distance_matrix:
    matrix_part:
    """
    _, label_to_cd_dis, _ = \
                    get_cluster_centroid(distance_matrix, matrix_part)

    inertia = sum(label_to_cd_dis.values())

    return inertia

def get_cluster_centroid(distance_matrix, matrix_part):
    """get cluster centroid, one of sample in the clusters 
    """ 
    label_to_index = {}
    for index, label in enumerate(matrix_part):
        if label not in label_to_index:
            label_to_index.update({label:[]})
        label_to_index[label].append(index)
    label_to_index = dict(sorted(label_to_index.items()))
    label_to_cd_index = {}      # centroid index
    label_to_cd_dis = {}        # centroid distance 
    for label in label_to_index:
        centroid_index = -1
        centroid_dis = math.inf
        for index in label_to_index[label]:
            tmp_dis = np.sum(distance_matrix[index, label_to_index[label]])
            if tmp_dis < centroid_dis:
                centroid_index = index
                centroid_dis = tmp_dis

        label_to_cd_index.update({label: centroid_index})
        label_to_cd_dis.update({label: centroid_dis})

    return label_to_cd_index, label_to_cd_dis, label_to_index

class ChamferDistanceL2(nn.Module):
    def __init__(self):
        super(ChamferDistanceL2, self).__init__()
        self.chamLoss = dist_chamfer_3D.chamfer_3DDist()

    def forward(self, prediction, gt, compress="mean"):
        dist1, dist2, _, _ = self.chamLoss(prediction, gt)
        if compress == "mean":
            loss_ptc_fine_atlas = torch.mean(dist1) + torch.mean(dist2)
            return loss_ptc_fine_atlas
        elif compress == "list":
            loss_ptc_fine_atlas = torch.mean(dist1, 1) + torch.mean(dist2, 1)
            return loss_ptc_fine_atlas


def pairwise_distances_torch(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    N = x.shape[0]
    dist = torch.pdist(x, p=2) ** 2
    matrix = torch.zeros(N, N)
    index = 0
    for i in range(0, N):
        for j in range(i+1, N):
            matrix[j, i] = dist[index]
            matrix[i, j] = matrix[j, i]
            index += 1
    
    matrix = matrix/x.shape[-1]

    return matrix


class PerceptualEncoder(torch.nn.Module):
    def __init__(self):
        """
        Implementation of perceptual loss from 
        Perceptual Losses for Real-Time Style Transfer and Super-Resolution
        https://arxiv.org/abs/1603.08155
        """
        super(PerceptualEncoder, self).__init__()
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.vgg = Vgg16()
    
    def forward(self, img):
        """In paper, content reconstruction loss use relu3_3 feature maps
        """
        with torch.set_grad_enabled(False):
            img = (img-self.mean) / self.std
            f_maps = self.vgg(img)
            return f_maps



if __name__ == "__main__":
    metric = torch.nn.MSELoss(reduction='none').cuda()
    a = torch.rand(200, 3000).cuda()
    dismat = compute_img_dismatrix_batch(a, a, metric, 32, torch.device('cuda'))

    dismat = dismat.cpu().numpy()
    print(np.allclose(dismat, dismat.T))
    test_idx1 = 23
    test_idx2 = 50

    test_metric = torch.nn.MSELoss().cuda()
    expected_ans = test_metric(a[[test_idx1]], a[[test_idx2]]).item()
    print("Expected", expected_ans)

    print("Actual", dismat[test_idx1, test_idx2])

