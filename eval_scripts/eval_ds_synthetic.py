import os
import sys
import signal
sys.path.append('../')
from training.gputracker import get_logger, DispatchThread
from os.path import join
import glob
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--gpus", nargs='+', type=int, default=[0, 1], help="ids of gpu to use")
args = parser.parse_args()
################################################################################################
############################### Input Dispersion Score of Synthetic Datasets ###################

results_base_dir = "eval/eval_results"
BASH_COMMAND_LIST = []
logger = get_logger(results_base_dir, 'schedule_eval_synthetic.log')
'''
split_list = ['train']
type_list = ['image', 'points']
json_list = ['image', 'shape']
metric_dic = {'image': 'mse', 'points': 'chamfer'}
SEED_LIST = '1'
nsample=-1
rsample=1
c_method_list = 'KMedoids'          # AP        Hierarchical
e_method_list = 'Inertia'           # Inertia   Inertia
cluster_k =     '2'                 # 2         2
perf_pc_list =  '0'                 # 4         0

cltsize_lst = [1, 5, 10, 15, 20, 25, 30, 35]

for json_type in json_list:
    for type in type_list:
        for cltsize in cltsize_lst:
            for split in split_list:
                BASH_COMMAND_LIST.append("python eval/input_ds_synthetic.py --SVR " \
                                                    f"--dir_name '{results_base_dir}/eval_ds_synthetic' " \
                                                    f"--rsample {rsample} " \
                                                    f"--nsample {nsample} " \
                                                    f"--type {type} " \
                                                    f"--mode 'object' " \
                                                    f"--seed_list {SEED_LIST} " \
                                                    f"--split {split} " \
                                                    f"--perceptual-max-pooling " \
                                                    f"--perf_pc_list {perf_pc_list} " \
                                                    f"--c_method {c_method_list} " \
                                                    f"--e_method {e_method_list} " \
                                                    f"--cluster_k {cluster_k} " \
                                                    f"--perceptual " \
                                                    f"--res_folder 'checkpoints_input' "
                                                    f"--metric {metric_dic[type]} " \
                                                    f"--data_base_dir 'dataset/synthetic_data/cubesphere_1000' " \
                                                    f"--train_json_file 'cluster_{json_type}_aug_2by10_cltsize{cltsize:02}.json' " \
                                                    f"--test_json_file 'test_interp_1000.json' " \
                                                    f"--val_json_file 'val_interp_1000.json' " \
                                                    f"--trained_exp_dir '' ")

split='test'
cltsize=1
type='points'
json_type='image'
BASH_COMMAND_LIST.append("python eval/input_ds_synthetic.py --SVR " \
                                    f"--dir_name '{results_base_dir}/eval_ds_synthetic' " \
                                    f"--rsample {rsample} " \
                                    f"--nsample {nsample} " \
                                    f"--type {type} " \
                                    f"--mode 'object' " \
                                    f"--seed_list {SEED_LIST} " \
                                    f"--split {split} " \
                                    f"--perceptual-max-pooling " \
                                    f"--perf_pc_list {perf_pc_list} " \
                                    f"--c_method {c_method_list} " \
                                    f"--e_method {e_method_list} " \
                                    f"--cluster_k {cluster_k} " \
                                    f"--perceptual " \
                                    f"--res_folder 'checkpoints_input' "
                                    f"--metric {metric_dic[type]} " \
                                    f"--data_base_dir 'dataset/synthetic_data/cubesphere_1000' " \
                                    f"--train_json_file 'cluster_{json_type}_aug_2by10_cltsize{cltsize:02}.json' " \
                                    f"--test_json_file 'test_interp_1000.json' " \
                                    f"--val_json_file 'val_interp_1000.json' " \
                                    f"--trained_exp_dir '' ")



dispatch_thread = DispatchThread("input ds evaluation", 
                 BASH_COMMAND_LIST, logger, gpu_m_th=8000, gpu_list=args.gpus, maxcheck=0)
# Start new Threads
dispatch_thread.start()
dispatch_thread.join()

import time
time.sleep(5)

logger.info("Exiting Main Thread")
'''
################################################################################################
############################### Output Dispersion Score of Synthetic Datasets ###################
BASH_COMMAND_LIST = []
log_base_dir = 'log/train_synthetic'
split = 'pred'

SEED_LIST = '1'
nsample=-1
rsample=1
c_method_list = 'KMedoids' # KMedoids KMedoids
e_method_list = 'Inertia'  # 
cluster_k =     '2'        #                
perf_pc_list =  '0'        #               
type = 'points'
pred_batch_size = 64
res_folder = 'checkpoints_pred'  # 'checkpoints_pred_lastepoch' checkpoints_pred_bestepoch
network = 'pseudo_network'                   #toydata_lastepoch  toydata_bestepoch
model = ('atlasnet', 'SPHERE', 1)

trained_exp_dir_lst = glob.glob(join(log_base_dir, "*Epoch*"))
trained_exp_dir_lst.sort()
for trained_exp_dir in trained_exp_dir_lst:
    json_type = trained_exp_dir.split('_')[4]
    cltsize = int(trained_exp_dir.split('_')[7][7:])
    BASH_COMMAND_LIST.append("python eval/output_ds_synthetic.py --SVR " \
                                        f"--dir_name '{results_base_dir}/eval_ds_synthetic' " \
                                        f"--rsample {rsample} " \
                                        f"--nsample {nsample} " \
                                        f"--type {type} " \
                                        f"--SVR " \
                                        f"--network {model[0]} " \
                                        f"--template_type {model[1]} " \
                                        f"--nb_primitives {model[2]} " \
                                        f"--bottleneck_size 1024 " \
                                        f"--hidden_neurons 512 " \
                                        f"--seed_list {SEED_LIST} " \
                                        f"--split {split} " \
                                        f"--pred_batch_size {pred_batch_size} " \
                                        f"--perf_pc_list {perf_pc_list} " \
                                        f"--c_method {c_method_list} " \
                                        f"--e_method {e_method_list} " \
                                        f"--cluster_k {cluster_k} " \
                                        f"--res_folder {res_folder} " \
                                        f"--reload_model_path '{trained_exp_dir}/checkpoint.pt'" \
                                        f"--metric 'chamfer' " \
                                        f"--data_base_dir 'dataset/synthetic_data/cubesphere_1000' " \
                                        f"--train_json_file 'cluster_{json_type}_aug_2by10_cltsize{cltsize:02}.json' " \
                                        f"--test_json_file 'test_interp_1000.json' " \
                                        f"--val_json_file 'val_interp_1000.json' " \
                                        f"--trained_exp_dir '{trained_exp_dir}' ")


dispatch_thread = DispatchThread("output ds evaluation", 
                 BASH_COMMAND_LIST, logger, gpu_m_th=8000, gpu_list=args.gpus, maxcheck=0)
# Start new Threads
dispatch_thread.start()
dispatch_thread.join()

import time
time.sleep(5)

logger.info("Exiting Main Thread")
