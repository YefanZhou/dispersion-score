"""Generate isotropic Gaussian blobs, 2D data, for metric demo.
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
"""
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import random
import sys
import numpy as np
import argparse
from os.path import join
sys.path.append("../../")
import auxiliary.my_utils as my_utils
parser = argparse.ArgumentParser(
    description='sum the integers at the command line')
parser.add_argument("--num_clusters", type=int, default=8,
    help='number of clusters')
parser.add_argument(
    '--num_points', type=int, default=200,
    help='')  
parser.add_argument(
    '--std_max_size', type=int, default=8,
    help='')
parser.add_argument(
    '--seed', type=int, default=2,
    help='')
parser.add_argument(
    '--side_length', type=int, default=25,
    help='')
parser.add_argument(
    '--dim', type=int, default=2,
    help='')
parser.add_argument(
    '--save_folder', type=str, default="clusters_data",
    help='')
parser.add_argument(
    '--log', default=sys.stdout, type=argparse.FileType('w'),
    help='the file where the sum should be written')
args = parser.parse_args()

my_utils.plant_seeds(args.seed)
num_of_clusters = args.num_clusters
color_list = ['b', 'g', 'r', 'c', 'm','y','k', 'pink', 'brown', 'cyan']
xs = np.random.randint(-1 * args.side_length, args.side_length, num_of_clusters)
ys = np.random.randint(-1 * args.side_length, args.side_length, num_of_clusters)

centers = [(x, y) for x, y in zip(xs, ys)]

for cluster_std in np.arange(1, args.std_max_size+1, 1):
    plt.figure(figsize=(5,5))
    #cluster_std = std_max_size * (np.random.random(num_of_clusters))
    cluster_std_array = np.array([cluster_std] * num_of_clusters)

    X, y = make_blobs(n_samples=args.num_points, cluster_std=cluster_std_array, centers=centers, n_features=args.dim, random_state=args.seed)

    for i in range(num_of_clusters):
        plt.scatter(X[y == i, 0], X[y == i, 1], color=color_list[i], s=80)
    
    plt.xticks([])
    plt.yticks([])
    data_fname = f'dim{args.dim}_ncls{args.num_clusters}_side{args.side_length}_nsample{args.num_points}_seed{args.seed}_std{cluster_std}.npz'
    plot_fname = data_fname.replace("npz", "png")
    plt.tight_layout()
    #plt.show()
    plt.savefig(join(args.save_folder, plot_fname), bbox_inches="tight")
    #np.savez(join(args.save_folder, data_fname), 
    #       data=X, label=y)



    



    