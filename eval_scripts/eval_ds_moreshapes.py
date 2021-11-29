import os
import sys
import signal
sys.path.append('../')
from os.path import join
from training.gputracker import get_logger, DispatchThread
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", nargs='+', type=int, default=[0, 1], help="ids of gpu to use")
args = parser.parse_args()

results_base_dir = "eval/eval_results"
trained_log_dir = "log/more_shapes"
logger = get_logger(results_base_dir, 'schedule_eval_more_shapes.log')
BASH_COMMAND_LIST = []
######################## Input DS of Training Image ##################################
#######################################################################################
SEED_LIST = '1'
nsample=-1
rsample=1
c_method_list = 'KMedoids'        
e_method_list = 'Inertia' 
cluster_k = '500'                              
perf_pc_list = '0'   
split='train'
mode='object'

BASH_COMMAND_LIST.append("python eval/input_ds.py --SVR " \
                                    f"--views_search " \
                                    f"--shapenet13 " \
                                    f"--dir_name '{results_base_dir}/more_shapes' " \
                                    f"--rsample {rsample} " \
                                    f"--nsample {nsample} " \
                                    f"--mode {mode} " \
                                    f"--type 'image' " \
                                    f"--seed_list {SEED_LIST} " \
                                    f"--nviews_train 1 " \
                                    f"--nviews_test 1 " \
                                    f"--split {split} " \
                                    f"--perceptual-max-pooling " \
                                    f"--perf_pc_list {perf_pc_list} " \
                                    f"--c_method {c_method_list} " \
                                    f"--e_method {e_method_list} " \
                                    f"--cluster_k {cluster_k} " \
                                    f"--perceptual " \
                                    f"--res_folder 'checkpoints_input' " \
                                    f"--metric 'mse' " \
                                    f"--trained_exp_dir '' ")
        
######################## Input DS of Training Points ##################
#######################################################################
MODE_LIST = ['object', 'viewer']
for mode in MODE_LIST:
    BASH_COMMAND_LIST.append("python eval/input_ds.py --SVR " \
                                        f"--views_search " \
                                        f"--shapenet13 " \
                                        f"--dir_name '{results_base_dir}/more_shapes' " \
                                        f"--rsample {rsample} " \
                                        f"--nsample {nsample} " \
                                        f"--mode {mode} " \
                                        f"--type 'points' " \
                                        f"--seed_list {SEED_LIST} " \
                                        f"--nviews_train 1 " \
                                        f"--nviews_test 1 " \
                                        f"--split {split} " \
                                        f"--perceptual-max-pooling " \
                                        f"--perf_pc_list {perf_pc_list} " \
                                        f"--c_method {c_method_list} " \
                                        f"--e_method {e_method_list} " \
                                        f"--cluster_k {cluster_k} " \
                                        f"--perceptual " \
                                        f"--res_folder 'checkpoints_input' " \
                                        f"--metric 'chamfer' " \
                                        f"--trained_exp_dir '' ")


####################### Output DS of Pred points ####################
#####################################################################
trained_folder_list = glob.glob(join(trained_log_dir, '*_seed*'))                       
split='pred'                                                       
for trained_folder in trained_folder_list:
    BASH_COMMAND_LIST.append("python eval/output_ds_viewer_inv.py --SVR " \
                                f"--views_search " \
                                f"--shapenet13 " \
                                f"--dir_name '{results_base_dir}/more_shapes' " \
                                f"--rsample {rsample} " \
                                f"--nsample {nsample} " \
                                f"--type 'points' " \
                                f"--nviews_train 1 " \
                                f"--nviews_test 1 " \
                                f"--seed_list 1 " \
                                f"--split 'pred' " \
                                f"--metric 'chamfer' " \
                                f"--perceptual_batch_size 32 " \
                                f"--pred_batch_size 64 " \
                                f"--dismat_batch_size 16 " \
                                f"--perf_pc_list {perf_pc_list} " \
                                f"--c_method {c_method_list} " \
                                f"--e_method {e_method_list} " \
                                f"--cluster_k {cluster_k} " \
                                f"--res_folder 'checkpoints_pred' " \
                                f"--trained_exp_dir {trained_folder} ")


dispatch_thread = DispatchThread("shapenet more imgs ds evaluations", 
                 BASH_COMMAND_LIST, logger, gpu_m_th=500, gpu_list=args.gpus, maxcheck=5)
# Start new Threads
dispatch_thread.start()
dispatch_thread.join()

import time
time.sleep(5)

logger.info("Exiting Main Thread")

