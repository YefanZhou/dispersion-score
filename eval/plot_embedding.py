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
parser.add_argument("--sample_num", type=int, default=20)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--perf_percent", type=int, default=50)
parser.add_argument("--if_dm_stats", action="store_true", default=False)
parser.add_argument("--embed_method", type=str, default='tsne', choices=['tsne', 'mds'])

args = parser.parse_args()
sample_num = args.sample_num
total_num = 3202
seed=args.seed
fontsize=10


#file_path = 'scripts/table_car/oc_vc_sscore/dm_part_sscore'
file_path = 'scripts/shapenet13/pts_testgt_dm_part_sscore'
plt_path = 'scripts/shapenet13/pts_testgt_dm_part_sscore/stats_plots'
graph_plt_path = f'scripts/shapenet13/pts_testgt_dm_part_sscore/{args.embed_method}_plots'
#graph_plt_path = f'scripts/table_car/oc_vc_sscore/dm_part_sscore/{args.embed_method}_plots'
Path(plt_path).mkdir(parents=True, exist_ok=True)
Path(graph_plt_path).mkdir(parents=True, exist_ok=True)

dm_keys = ['lower_quat', 'up_quat', 'mean', 'min', 'max', 'median', 'std']

checkpoints = glob.glob(os.path.join(file_path, f'*_{sample_num}.npz'))
checkpoints.sort()
print(checkpoints)
cat_path = [f.replace('dm_part_sscore', 'cat_fetch') for f in checkpoints]
cat_path = [f.replace('points', 'category') for f in cat_path]
obj_stats = np.load(checkpoints[0], allow_pickle=True)
view_stats = np.load(checkpoints[1], allow_pickle=True)
obj_cat = obj_stats[str(seed)].item()['class_labels']
view_cat = view_stats[str(seed)].item()['class_labels']
#obj_cat = np.load(cat_path[0], allow_pickle=True)[str(seed)]
#view_cat = np.load(cat_path[1], allow_pickle=True)[str(seed)]

obj_label = obj_stats[str(seed)].item()[str(args.perf_percent)]['label']
obj_sscore = obj_stats[str(seed)].item()[str(args.perf_percent)]['sscore']
view_label = view_stats[str(seed)].item()[str(args.perf_percent)]['label']
view_sscore = view_stats[str(seed)].item()[str(args.perf_percent)]['sscore']
# obj_label = obj_stats[str(seed)].item()['label']
# obj_sscore = obj_stats[str(seed)].item()['sscore']
# view_label = view_stats[str(seed)].item()['label']
# view_sscore = view_stats[str(seed)].item()['sscore']

obj_dismat = obj_stats[str(seed)].item()['dm']
view_dismat = view_stats[str(seed)].item()['dm']

if args.embed_method == 'tsne':
    obj_embedding = TSNE(n_components=2, metric='precomputed', random_state=args.seed, n_jobs=10).fit_transform(obj_dismat)
    view_embedding = TSNE(n_components=2, metric='precomputed', random_state=args.seed, n_jobs=10).fit_transform(view_dismat)
elif args.embed_method == 'mds':
    obj_embedding = MDS(n_components=2, metric=True, n_jobs=10, random_state=args.seed, dissimilarity='precomputed').fit_transform(obj_dismat)
    view_embedding = MDS(n_components=2, metric=True, n_jobs=10, random_state=args.seed, dissimilarity='precomputed').fit_transform(view_dismat)

fig, axes = plt.subplots(2, 2, figsize=(10,10))
mapper, color_cat = label_to_colr_mapper(obj_cat, mpl.cm.gist_rainbow)   #coolwarm
axes[0, 0].scatter(obj_embedding[:, 0], obj_embedding[:, 1], s=10, c=[mapper.to_rgba(i) for i in color_cat.values()])
#axes[0, 0].set_title("Object Center Semantic Label")


mapper, color_cat = label_to_colr_mapper(view_cat, mpl.cm.gist_rainbow)   #coolwarm
axes[0, 1].scatter(view_embedding[:, 0], view_embedding[:, 1], s=10, c=[mapper.to_rgba(i) for i in color_cat.values()])
#axes[0, 1].set_title("Viewer Center Semantic Label")


mapper, color_cat = label_to_colr_mapper(obj_label, mpl.cm.gist_rainbow)
obj_nclusters = max(color_cat.values())+1
axes[1, 0].scatter(obj_embedding[:, 0], obj_embedding[:, 1], s=10, c=[mapper.to_rgba(i) for i in color_cat.values()])
#axes[1, 0].set_title("Object Center Clustering Label")

mapper, color_cat = label_to_colr_mapper(view_label, mpl.cm.gist_rainbow)
view_nclusters = max(color_cat.values())+1
axes[1, 1].scatter(view_embedding[:, 0], view_embedding[:, 1], s=10, c=[mapper.to_rgba(i) for i in color_cat.values()])
#axes[1, 1].set_title("Viewer Center Clustering Label")

axes[0,0].set_title(f"{args.embed_method} OC Semantic label")
axes[0,1].set_title(f"{args.embed_method} VC Semantic label")
axes[1,0].set_title(f"{args.embed_method} OC Cluster Label Affinity prop nclusters: {obj_nclusters} \n silhouette: {obj_sscore:.3f}")
axes[1,1].set_title(f"{args.embed_method} VC Cluster Label Affinity prop nclusters: {view_nclusters} \n silhouette: {view_sscore:.3f}")

plt.suptitle(f"sample number: {sample_num}, Seed: {seed}, Pref perc: {args.perf_percent} %")
plt.tight_layout()
#print(join(graph_plt_path, f'{sample_num}_seed{seed}_prefpc{args.perf_percent}.png'))

plt.savefig(join(graph_plt_path, f'{sample_num}_seed{seed}_prefpc{args.perf_percent}.png'))
#plt.show()

