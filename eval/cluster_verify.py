import os
import sys
from os.path import join
import numpy as np
sys.path.append(join(os.path.dirname(os.path.abspath(__file__)), "../"))
from eval.metric import cluster_eval


path = "object_pred_points_trainnv01_testnv01_8762_210205_0722_object_optimize1_pretrainedTrue_decay30.npz"

data = np.load(os.path.join('scripts/shapenet13_benchmark/checkpoints_pred', path), allow_pickle=True)


data  = data['1'].item()
print(data['dm'])


# print(data['AP_Inertia_k0p10']['label'])
# print(data['AP_Inertia_k0p10']['label_stats'])

# print(data['AP_Inertia_k0p50']['label'])
# print(data['AP_Inertia_k0p50']['label_stats'])

score, part_label = cluster_eval(c_method='AP', e_method="Inertia", distance_matrix=data['dm'], 
        seed=1, n_cluster=0, pc=10)

print(part_label)