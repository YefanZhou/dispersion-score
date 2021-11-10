import os
import sys
import signal
sys.path.append('../')
from os.path import join
from training.gputracker import get_logger
import glob

##########################Output DS of Views Search #############################
#################################################################################
BASH_COMMAND_LIST = []
nsample=-1
rsample=1
mode='viewer'
results_base_dir = "eval/eval_results"
trained_log_dir = "log/nviews"
logger = get_logger(results_base_dir, 'schedule_eval_nviews.log')
trained_folder_lst = glob.glob(join(trained_log_dir, '*'))


logger = get_logger("./", 'shapenet13_nviews.log')
                       
c_method_list = 'KMedoids'        
e_method_list = 'Inertia' 
cluster_k = '500'                              
perf_pc_list = '0'                             
split='pred'
SAMPLE_LIST = [(0, 1)]                                    

for trained_folder in trained_folder_lst:
    BASH_COMMAND_LIST.append("python eval/output_ds.py --SVR " \
                                f"--views_search " \
                                f"--shapenet13 " \
                                f"--dir_name {results_base_dir}/nviews " \
                                f"--rsample {rsample} " \
                                f"--nsample {nsample} " \
                                f"--mode {mode} " \
                                f"--type 'points' " \
                                f"--nviews_train 1 " \
                                f"--nviews_test 1 " \
                                f"--seed_list 1 " \
                                f"--split {split} " \
                                f"--metric 'chamfer' " \
                                f"--perceptual_batch_size 32 " \
                                f"--pred_batch_size 64 " \
                                f"--dismat_batch_size 16 " \
                                f"--perf_pc_list {perf_pc_list} " \
                                f"--c_method {c_method_list} " \
                                f"--e_method {e_method_list} " \
                                f"--cluster_k {cluster_k} " \
                                f"--res_folder 'checkpoints_pred' "
                                f"--trained_exp_dir {trained_folder} ")

BASH_COMMAND_LIST.sort()

for command in BASH_COMMAND_LIST[-1:]:
    
    logger.info(f"Launching Experiments: {command}")
    os.system(command)
    # sleep 5 seconds 
    code = os.system('sleep 5')

    if code == signal.SIGINT:
        logger.info('Keyboard Interpret')
        break