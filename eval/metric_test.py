import logging
import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
sys.path.append("../")
from eval.metric import inertia_ap, get_cluster_centroid, compute_ptcloud_dismatrix_batch, \
                        compute_ptcloud_dismatrix, ChamferDistanceL2
from scipy.special import gammainc
from sklearn.metrics import pairwise_distances


##Test func: inertia_ap
def test_get_cluster_centroid():
    r= 30.*np.sqrt(np.random.rand(1000))
    phi = 2. * np.pi * np.random.rand(1000)

    x = r * np.cos(phi)
    y = r * np.sin(phi)
    cords = np.vstack((x, y))
    cords = np.transpose(cords)

    dis_mat = pairwise_distances(cords)
    label = [0] * dis_mat.shape[0]
    label_to_cd_index, label_to_cd_dis = get_cluster_centroid(dis_mat, label)

    centroid = cords[label_to_cd_index[0]]

    plt.figure()
    plt.plot(x, y, 'o', c='b')
    plt.plot(centroid[0], centroid[1], 'o', c='r')
    plt.show()

def test_batch_distance_matrix():
    device = torch.device("cuda")
    chamfer = ChamferDistanceL2()
    logger = logging.getLogger("INFO")
    N = 8762
    num_points = 2500
    dim = 3
    batch_size = 32
    data = torch.rand(N, num_points, dim)
    data = data.to(device)

    batch_dismat = compute_ptcloud_dismatrix_batch(data, data, chamfer, batch_size, device, logger)
    batch_dismat = batch_dismat.cpu().numpy()
    symm = np.allclose(batch_dismat, batch_dismat.T)
    print(f"Batch distance matrix is symmetric: {symm}")

    loop_dismat = compute_ptcloud_dismatrix(data, data, chamfer, logger)
    loop_dismat = loop_dismat.cpu().numpy()
    sanity_check = np.allclose(loop_dismat, batch_dismat)

    print(f"Sanity Check Pass: {sanity_check}")

    """
    def compute_ptcloud_dismatrix_batch
    time: 26s

    def compute_ptcloud_dismatrix
    time: 03:29

    Batch distance matrix is symmetric: True
    Sanity Check Pass: True
    """






if __name__ == "__main__":
    #test_get_cluster_centroid()
    test_batch_distance_matrix()
