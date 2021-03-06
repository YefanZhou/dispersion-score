import os
import sys
import argparse
sys.path.append('../')

from training.gputracker import get_logger, DispatchThread

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", nargs='+', type=int, default=[0, 1], help="ids of gpu to use")
args = parser.parse_args()
###################################################################################################
#############################ShapeNet 13 AtlasNet Views Search FIXED EVAL ######################################

logger = get_logger('log', 'schedule_views_fixed_eval.log')
SEED_LIST=[1, 2, 3]
MODE_OPTIONS=['viewer']
MODEL_OPTIONS=[('atlasnet', 'SPHERE', 1)]
DECAY_OPTIONS=[(120, 90, 110, 115)]
TRAIN_VIEW_OPTIONS=[1, 3, 6, 9, 12, 15, 18] #21, 23, 2, 4, 5
TEST_VIEWS=1
## Train
BASH_COMMAND_LIST = []
for SEED in SEED_LIST:
    for decay_idx, decay in enumerate(DECAY_OPTIONS):
        for mode_idx, mode in enumerate(MODE_OPTIONS):
            for model_idx, model in enumerate(MODEL_OPTIONS):
                for train_idx, train_nviews in enumerate(TRAIN_VIEW_OPTIONS):
                    BASH_COMMAND_LIST.append(f"python train.py --shapenet13 --SVR " \
                                            f"--dir_name 'log/nviews' " \
                                            f"--tbd_dir 'log/tensorboard/nviews' " \
                                            f"--network {model[0]} " \
                                            f"--template_type {model[1]} " \
                                            f"--nb_primitives {model[2]} " \
                                            f"--batch_size 64 " \
                                            f"--batch_size_test 64 " \
                                            f"--workers 8 " \
                                            f"--lrate 1e-3 " \
                                            f"--nepoch {decay[0]} " \
                                            f"--lr_decay_1 {decay[1]} " \
                                            f"--lr_decay_2 {decay[2]} " \
                                            f"--lr_decay_3 {decay[3]} " \
                                            f"--manual_seed {SEED} " \
                                            f"--views_search " \
                                            f"--mode {mode} " \
                                            f"--nviews_train {train_nviews} " \
                                            f"--nviews_test {TEST_VIEWS} " \
                                            f"--description '{mode}_{model[0]}{model[1]}{model[2]}_decy{decay_idx}_trainnv{train_nviews}_testnv{TEST_VIEWS}_seed{SEED}_fixed_eval'")


dispatch_thread = DispatchThread("shapenet more imgs ds evaluations", 
                 BASH_COMMAND_LIST, logger, gpu_m_th=500, gpu_list=args.gpus, maxcheck=5)
# Start new Threads
dispatch_thread.start()
dispatch_thread.join()

import time
time.sleep(5)

logger.info("Exiting Main Thread")