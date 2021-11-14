import os
import sys
import signal
import argparse
sys.path.append('../')
from training.gputracker import get_logger, DispatchThread

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", nargs='+', type=int, default=[0, 1], help="ids of gpu to use")
args = parser.parse_args()
###################################################################################################
#############################ShapeNet 13 Rendering Yaw Limit ######################################
BASH_COMMAND_LIST = []
logger = get_logger('log', 'schedule_more_imgs.log')
MODEL_OPTIONS=[('atlasnet', 'SPHERE', 1)]  
DECAY_OPTIONS=[(120, 90, 110, 115)]
MODE_OPTIONS=['object']      
SEED_LIST = [1, 2, 3]              
BATCH_SIZE = 64   
HIDDEN_NEURONS = 512
BOTTLENECK = 1024
YAWRANGE_LIST = [0, 15, 30, 45, 60, 75, 90] 

for SEED in SEED_LIST:
    for mode in MODE_OPTIONS:
        for YAWRANGE in YAWRANGE_LIST:
            for decay in DECAY_OPTIONS:
                for model_idx, model in enumerate(MODEL_OPTIONS):
                    BASH_COMMAND_LIST.append(f"python train.py --SVR " \
                                            f"--shapenet13 " \
                                            f"--rendering_root_dir 'dataset/data/ShapeNetV1RenderingAnglelimit/ShapeNetV1RenderingSampleYaw{YAWRANGE:03}/image' " \
                                            f"--yawrange {YAWRANGE} " \
                                            f"--dir_name 'log/more_imgs' " \
                                            f"--mode {mode} " \
                                            f"--network {model[0]} " \
                                            f"--template_type {model[1]} " \
                                            f"--nb_primitives {model[2]} " \
                                            f"--batch_size {BATCH_SIZE} " \
                                            f"--batch_size_test {BATCH_SIZE} " \
                                            f"--workers 4 " \
                                            f"--lrate 1e-3 " \
                                            f"--tbd_dir 'log/tensorboard/more_imgs' " \
                                            f"--nepoch {decay[0]} " \
                                            f"--lr_decay_1 {decay[1]} " \
                                            f"--lr_decay_2 {decay[2]} " \
                                            f"--lr_decay_3 {decay[3]} " \
                                            f"--manual_seed {SEED} " \
                                            f"--anglelimit_search " \
                                            f"--bottleneck_size {BOTTLENECK} " \
                                            f"--hidden_neurons {HIDDEN_NEURONS} " \
                                            f"--description 'more_imgs_yawrange{YAWRANGE:03}_{mode}_{model[0]}{model[1]}{model[2]}_seed{SEED}'")


dispatch_thread = DispatchThread("synthetic dataset training", 
                 BASH_COMMAND_LIST, logger, gpu_m_th=500, gpu_list=args.gpus, maxcheck=5)
# Start new Threads
dispatch_thread.start()
dispatch_thread.join()

import time
time.sleep(5)

logger.info("Exiting Main Thread")