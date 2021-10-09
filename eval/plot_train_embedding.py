import glob
import os
import numpy as np
from eval_utils import array_to_dic, verbose_dic, dic_to_box, list_to_indexdic, label_to_colr_mapper
from metric import silhouette_score_ap
import matplotlib.pyplot as plt
from os.path import join
import matplotlib as mpl
import networkx as nx
from pygraphviz import *
import argparse
from sklearn.manifold import TSNE, MDS
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--sample_num", type=int, default=2000)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--perf_percent", type=int, default=50)
parser.add_argument("--if_dm_stats", action="store_true", default=False)
parser.add_argument("--embed_method", type=str, default='tsne', choices=['tsne', 'mds'])

args = parser.parse_args()
sample_num = args.sample_num
total_num = 3202
seed=args.seed
fontsize=10


file_path = 'scripts/shapenet13_pred/dm_part_sscore'
graph_plt_path = f'scripts/shapenet13_pred/dm_part_sscore/{args.embed_method}_plots'
#graph_plt_path = f'scripts/table_car/oc_vc_sscore/dm_part_sscore/{args.embed_method}_plots'

Path(graph_plt_path).mkdir(parents=True, exist_ok=True)

dm_keys = ['lower_quat', 'up_quat', 'mean', 'min', 'max', 'median', 'std']

checkpoints = glob.glob(os.path.join(file_path, f'*.npz'))
checkpoints.sort()
for idx, cp in enumerate(checkpoints):
    view_stats = np.load(cp, allow_pickle=True)

    view_cat = view_stats[str(seed)].item()['class_labels']

    view_label = view_stats[str(seed)].item()[str(args.perf_percent)]['label']
    view_sscore = view_stats[str(seed)].item()[str(args.perf_percent)]['sscore']

    view_dismat = view_stats[str(seed)].item()['dm']

    if args.embed_method == 'tsne':
        #obj_embedding = TSNE(n_components=2, metric='precomputed', random_state=args.seed, n_jobs=10).fit_transform(obj_dismat)
        view_embedding = TSNE(n_components=2, metric='precomputed', random_state=args.seed, n_jobs=10).fit_transform(view_dismat)
    elif args.embed_method == 'mds':
        #obj_embedding = MDS(n_components=2, metric=True, n_jobs=10, random_state=args.seed, dissimilarity='precomputed').fit_transform(obj_dismat)
        view_embedding = MDS(n_components=2, metric=True, n_jobs=10, random_state=args.seed, dissimilarity='precomputed').fit_transform(view_dismat)

    fig, axes = plt.subplots(1, 2, figsize=(10,5))

    mapper, color_cat = label_to_colr_mapper(view_cat, mpl.cm.gist_rainbow)   #coolwarm
    axes[0].scatter(view_embedding[:, 0], view_embedding[:, 1], s=10, c=[mapper.to_rgba(i) for i in color_cat.values()])


    mapper, color_cat = label_to_colr_mapper(view_label, mpl.cm.gist_rainbow)
    view_nclusters = max(color_cat.values())+1
    axes[1].scatter(view_embedding[:, 0], view_embedding[:, 1], s=10, c=[mapper.to_rgba(i) for i in color_cat.values()])


    axes[0].set_title(f"{args.embed_method} VC Semantic label")
    axes[1].set_title(f"{args.embed_method} VC Cluster Label Affinity prop nclusters: {view_nclusters}")

    plt.suptitle(f"sample number: {sample_num}, Seed: {seed}, Pref perc: {args.perf_percent} %")
    plt.tight_layout()

   #plt.savefig(join(graph_plt_path, f'{sample_num}_seed{seed}_prefpc{args.perf_percent}.png'))
    plt.show()
#plt.show()

