import os
import sys
import signal
sys.path.append('../')
from os.path import join
from training.gputracker import get_logger
import glob

results_base_dir = "eval/eval_results"
trained_log_dir = "log/more_imgs"

########################Input DS of Training Image############
##############################################################
BASH_COMMAND_LIST = []
logger = get_logger(results_base_dir, 'schedule_eval_moreimgs.log')
mode_list=["object"]
split_list=['train']                             # 'train'
PROB=0.5
SEED_LIST = '1'
YAWRANGE_LIST = [0, 15, 30, 45, 60, 75, 90]      # 
nsample=-1
rsample=1
c_method_list = 'KMedoids'  #KMedoids KMedoids
e_method_list = 'Inertia' #  Inertia Inertia
cluster_k = '500'  # 
perf_pc_list = '0' #  

for YAWRANGE in YAWRANGE_LIST:
    for mode in mode_list:
        for split in split_list:
            BASH_COMMAND_LIST.append("python eval/input_ds_aug.py --SVR " \
                                                f"--anglelimit_search " \
                                                f"--shapenet13 " \
                                                f"--dir_name '{results_base_dir}/more_imgs' " \
                                                f"--rsample {rsample} " \
                                                f"--nsample {nsample} " \
                                                f"--mode {mode} " \
                                                f"--type 'image' " \
                                                f"--yawrange {YAWRANGE} " \
                                                f"--rendering_root_dir 'dataset/data/ShapeNetV1RenderingAnglelimit/ShapeNetV1RenderingSampleYaw{YAWRANGE:03}/image' " \
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
                                                f"--res_folder 'checkpoints_{rsample}' "
                                                f"--metric 'mse' " \
                                                f"--trained_exp_dir '' ")
        
#######################Input DS of Training Points################
##################################################################

TYPE = 'points'
split = 'train'
mode = 'object'
YAWRANGE = 0
BASH_COMMAND_LIST.append("python eval/input_ds_aug.py --SVR " \
                                    f"--anglelimit_search " \
                                    f"--shapenet13 " \
                                    f"--dir_name '{results_base_dir}/more_imgs' " \
                                    f"--rsample {rsample} " \
                                    f"--nsample {nsample} " \
                                    f"--mode {mode} " \
                                    f"--type {TYPE} " \
                                    f"--yawrange {YAWRANGE} " \
                                    f"--rendering_root_dir 'dataset/data/ShapeNetV1RenderingAnglelimit/ShapeNetV1RenderingSampleYaw{YAWRANGE:03}/image' " \
                                    f"--seed_list {SEED_LIST} " \
                                    f"--nviews_train 1 " \
                                    f"--nviews_test 1 " \
                                    f"--split {split} " \
                                    f"--perf_pc_list {perf_pc_list} " \
                                    f"--c_method {c_method_list} " \
                                    f"--e_method {e_method_list} " \
                                    f"--cluster_k {cluster_k} " \
                                    f"--perceptual " \
                                    f"--res_folder 'checkpoints_{rsample}' "
                                    f"--metric 'chamfer' " \
                                    f"--trained_exp_dir '' ")

########################Ouput DS of Predicted Points################
#####################################################################
## Object centered
trained_folder_lst = glob.glob(join(trained_log_dir, '*'))
YAWRANGE = 90
split='pred'

for trained_folder in trained_folder_lst:
    BASH_COMMAND_LIST.append("python eval/output_ds_aug.py --SVR " \
                                        f"--anglelimit_search " \
                                        f"--shapenet13 " \
                                        f"--dir_name '{results_base_dir}/more_imgs' " \
                                        f"--rsample {rsample} " \
                                        f"--nsample {nsample} " \
                                        f"--type 'points' " \
                                        f"--yawrange {YAWRANGE} " \
                                        f"--rendering_root_dir 'dataset/data/ShapeNetV1RenderingAnglelimit/ShapeNetV1RenderingSampleYaw{YAWRANGE:03}/image' " \
                                        f"--seed_list {SEED_LIST} " \
                                        f"--nviews_train 1 " \
                                        f"--nviews_test 1 " \
                                        f"--split {split} " \
                                        f"--perf_pc_list {perf_pc_list} " \
                                        f"--c_method {c_method_list} " \
                                        f"--e_method {e_method_list} " \
                                        f"--cluster_k {cluster_k} " \
                                        f"--res_folder 'checkpoints_pred' "
                                        f"--metric 'chamfer' " \
                                        f"--trained_exp_dir {trained_folder} ")


for command in BASH_COMMAND_LIST:
    
    logger.info(f"Launching Experiments: {command}")
    os.system(command)
    # sleep 5 seconds 
    code = os.system('sleep 5')

    if code == signal.SIGINT:
        logger.info('Keyboard Interpret')
        break