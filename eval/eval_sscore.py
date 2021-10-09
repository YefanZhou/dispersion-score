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
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--sample_num", type=int, default=20)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--perf_percent", type=int, default=50)
parser.add_argument("--if_dm_stats", action="store_true", default=False)

args = parser.parse_args()
sample_num = args.sample_num
total_num = 3202
seed=args.seed
fontsize=10


#file_path = 'scripts/table_car/oc_vc_sscore/dm_part_sscore'
file_path = 'scripts/shapenet13/pts_testgt_dm_part_sscore'
plt_path = 'scripts/table_car/oc_vc_sscore/dm_part_sscore/stats_plots'
#graph_plt_path = 'scripts/table_car/oc_vc_sscore/dm_part_sscore/graph_plots_shapenet13'
graph_plt_path = 'scripts/shapenet13/pts_testgt_dm_part_sscore/graph_plots_shapenet13'
Path(plt_path).mkdir(parents=True, exist_ok=True)
Path(graph_plt_path).mkdir(parents=True, exist_ok=True)

dm_keys = ['lower_quat', 'up_quat', 'mean', 'min', 'max', 'median', 'std']

checkpoints = glob.glob(os.path.join(file_path, f'*_{sample_num}.npz'))
checkpoints.sort()

cat_path = [f.replace('dm_part_sscore', 'cat_fetch') for f in checkpoints]
cat_path = [f.replace('points', 'category') for f in cat_path]
obj_stats = np.load(checkpoints[0], allow_pickle=True)
view_stats = np.load(checkpoints[1], allow_pickle=True)
obj_cat = obj_stats[str(seed)].item()['class_labels']
view_cat = view_stats[str(seed)].item()['class_labels']
#obj_cat = np.load(cat_path[0], allow_pickle=True)
#view_cat = np.load(cat_path[1], allow_pickle=True)


if args.if_dm_stats:
    print(obj_stats['ss_stats'],  obj_stats['sample_num'], 
            obj_stats['mode'], obj_stats['type'], obj_stats['split'])
    print(view_stats['ss_stats'],  view_stats['sample_num'], 
            view_stats['mode'], view_stats['type'], view_stats['split'])


    oc_dm_stats_dic = array_to_dic(obj_stats[str(seed)].item()['dm_stats'], dm_keys)
    oc_label_stats = array_to_dic(obj_stats[str(seed)].item()['label_stats'], [])
    print('object', obj_stats[str(seed)].item()['sscore'], verbose_dic(oc_dm_stats_dic))
    print(oc_label_stats)


    vc_dm_stats_dic = array_to_dic(view_stats[str(seed)].item()['dm_stats'], dm_keys)
    vc_label_stats = array_to_dic(view_stats[str(seed)].item()['label_stats'], [])
    print('viewer', view_stats[str(seed)].item()['sscore'], verbose_dic(vc_dm_stats_dic))
    print(vc_label_stats)

    fig, axes = plt.subplots(2, 2, figsize=(fontsize, fontsize))
    ymax = max(list(oc_label_stats.values()) + list(vc_label_stats.values()))
    ymin = min(list(oc_label_stats.values()) + list(vc_label_stats.values()))

    axes[0, 0].bar(list(oc_label_stats.keys()), oc_label_stats.values(), color='g')
    axes[0, 0].set_ylim([ymin, ymax])
    axes[0, 0].set_title("OC Partition Label Distribution", fontsize=fontsize)
    axes[1, 0].bxp(dic_to_box(oc_dm_stats_dic), showfliers=False)
    axes[1, 0].set_ylabel("Chamfer Distance", fontsize=fontsize)
    axes[1, 0].set_title("OC Distance Matrix Distribution", fontsize=fontsize)


    axes[0, 1].bar(list(vc_label_stats.keys()), vc_label_stats.values(), color='g')
    axes[0, 1].set_ylim([ymin, ymax])
    axes[0, 1].set_title("VC Partition Label Distribution", fontsize=fontsize)
    axes[1, 1].bxp(dic_to_box(vc_dm_stats_dic), showfliers=False)
    axes[1, 1].set_ylabel("Chamfer Distance", fontsize=fontsize)
    axes[1, 1].set_title("VC Distance Matrix Distribution", fontsize=fontsize)
    plt.suptitle(f"Sample number: {sample_num} / {total_num}, perf percetile: {obj_stats[str(seed)].item()['perf_percent']}% Seed {seed}\n"
                f"OC silhouette: {obj_stats[str(seed)].item()['sscore']:.3f} \n" + 
                f"VC silhouette: {view_stats[str(seed)].item()['sscore']:.3f}", fontsize=fontsize)
    plt.savefig(join(plt_path, f'{sample_num}_seed{seed}.png'))
# plt.show()

#################################################################################
else:
    if args.perf_percent != 50:
        ss, matrix_part, part_preference = silhouette_score_ap(obj_stats[str(args.seed)].item()['dm'], \
                                                seed=args.seed, pc=args.perf_percent)
        obj_label = matrix_part
        obj_sscore = ss
        ss, matrix_part, part_preference = silhouette_score_ap(view_stats[str(args.seed)].item()['dm'], \
                                                seed=args.seed, pc=args.perf_percent)
        view_label = matrix_part
        view_sscore = ss
        
    else:
        obj_label = obj_stats[str(seed)].item()[str(args.perf_percent)]['label']
        obj_sscore = obj_stats[str(seed)].item()[str(args.perf_percent)]['sscore']
        view_label = view_stats[str(seed)].item()[str(args.perf_percent)]['label']
        view_sscore = view_stats[str(seed)].item()[str(args.perf_percent)]['sscore']

    fig, axes = plt.subplots(2, 2, figsize=(10,10))
    obj_dismat = obj_stats[str(seed)].item()['dm']
    #obj_cat = obj_cat[str(seed)]
    view_dismat = view_stats[str(seed)].item()['dm']
    #view_cat = view_cat[str(seed)]

    obj_G = nx.from_numpy_matrix(obj_dismat)
    view_G = nx.from_numpy_matrix(view_dismat)
    weight_list = []
    for (u, v, weight) in obj_G.edges.data('weight'):
        weight_list.append(weight)
    for (u, v, weight) in view_G.edges.data('weight'):
        weight_list.append(weight)
    min_w = min(weight_list)
    max_w = max(weight_list)

    # for (u, v, weight) in G.edges.data('weight'):
    #     print(f"Edge {u} -> {v} weights {weight}.")

    A = AGraph()
    #mapper, color_cat = label_to_colr_mapper(obj_cat, mpl.cm.coolwarm)
    mapper, color_cat = label_to_colr_mapper(obj_cat, mpl.cm.gist_rainbow)
    A.add_nodes_from = list(obj_G.nodes)
    for (u, v, weight) in obj_G.edges.data('weight'):
        #A.add_edge(u, v,len=round(((weight - min_w) + 0.0001)/(max_w - min_w), 4))
        A.add_edge(u, v,len=round(weight, 4))
    
    obj_G = nx.nx_agraph.from_agraph(A)
    pos = nx.nx_agraph.graphviz_layout(obj_G)
    nx.draw_networkx_nodes(obj_G, pos, node_size=20, node_color=[mapper.to_rgba(i) 
                        for i in color_cat.values()], ax=axes[0, 0])

    mapper, color_cat = label_to_colr_mapper(obj_label, mpl.cm.gist_rainbow)
    cluster_edges = []
    for tuple in obj_G.edges:
        if color_cat[int(tuple[0])] == color_cat[int(tuple[1])]:
            cluster_edges.append(tuple)

    nx.draw_networkx_nodes(obj_G, pos, node_size=20, node_color=[mapper.to_rgba(i) 
                        for i in color_cat.values()], ax=axes[1, 0])
    nx.draw_networkx_edges(obj_G, pos, cluster_edges, width=0.3, alpha=0.4, ax=axes[1, 0]) 

    obj_nclusters = max(color_cat.values())+1
    ###################################################################################################

    color_cat = list_to_indexdic(view_cat)
    A = AGraph()
    #mapper, color_cat = label_to_colr_mapper(view_cat, mpl.cm.coolwarm)
    mapper, color_cat = label_to_colr_mapper(view_cat, mpl.cm.gist_rainbow)
    A.add_nodes_from = list(view_G.nodes)
    for (u, v, weight) in view_G.edges.data('weight'):
        #A.add_edge(u, v,len=round(((weight - min_w) + 0.0001) /(max_w - min_w), 4))
        A.add_edge(u, v,len=round(weight, 4))
        
    view_G = nx.nx_agraph.from_agraph(A)
    pos = nx.nx_agraph.graphviz_layout(view_G)
    nx.draw_networkx_nodes(view_G, pos, node_size=20, node_color=[mapper.to_rgba(i) 
                        for i in color_cat.values()], ax=axes[0, 1])

    mapper, color_cat = label_to_colr_mapper(view_label, mpl.cm.gist_rainbow)
    cluster_edges = []

    for tuple in view_G.edges:
        if color_cat[int(tuple[0])] == color_cat[int(tuple[1])]:
            cluster_edges.append(tuple)

    nx.draw_networkx_nodes(view_G, pos, node_size=20, node_color=[mapper.to_rgba(i) 
                        for i in color_cat.values()], ax=axes[1, 1])

    nx.draw_networkx_edges(view_G, pos, cluster_edges, width=0.3, alpha=0.4, ax=axes[1, 1]) 
    view_nclusters = max(color_cat.values())+1

    #axes[0,0].set_title("OC Semantic label, red: Car blue:Table")
    axes[0,0].set_title("OC Semantic label")
    #axes[0,1].set_title("VC Semantic label, red: Car blue:Table")
    axes[0,1].set_title("VC Semantic label")
    axes[1,0].set_title(f"OC Cluster Label Affinity prop nclusters: {obj_nclusters} \n silhouette: {obj_sscore:.3f}")
    axes[1,1].set_title(f"VC Cluster Label Affinity prop nclusters: {view_nclusters} \n silhouette: {view_sscore:.3f}")

    #plt.tight_layout()
    plt.suptitle(f"sample number: {sample_num}, Seed: {seed}, Pref perc: {args.perf_percent} %")
    plt.tight_layout()
    plt.savefig(join(graph_plt_path, f'{sample_num}_seed{seed}_prefpc{args.perf_percent}.png'))
    #plt.show()










#pos = nx.spring_layout(G, seed=1)
#edge_widths = [w for (*edge, w) in G.edges.data('weight')]
#nx.draw_networkx_nodes(G, pos, node_size=10, ax=axes[0])
#nx.draw(G, pos, width=edge_widths, connectionstyle='arc3, rad=.15', node_size=10, ax=axes[0])
# axes[0].set_xlim([-1.5, 1.5])
# axes[0].set_ylim([-1.5, 1.5])
# axes[0].tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

# dismat = view_stats[str(seed)].item()['dm']
# label = view_stats[str(seed)].item()['label']
# G = nx.from_numpy_matrix(dismat)
# pos = nx.spring_layout(G)
# edge_widths = [w for (*edge, w) in G.edges.data('weight')]
# #nx.draw(G, node_size=4, ax=axes[1])
# nx.draw(G, pos, width=edge_widths, connectionstyle='arc3, rad=.15', node_size=10, ax=axes[1])
# axes[1].set_xlim([-1.5, 1.5])
# axes[1].set_ylim([-1.5, 1.5])
# axes[1].tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
# plt.show()

