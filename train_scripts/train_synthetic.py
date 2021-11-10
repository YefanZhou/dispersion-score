import os
import sys
import signal
sys.path.append('../')
from training.gputracker import get_logger, DispatchThread
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", nargs='+', type=int, default=[0, 1], help="ids of gpu to use")
args = parser.parse_args()

BASH_COMMAND_LIST  = [] 
logger = get_logger('log', 'schedule_train_synthetic.log')
model = ('atlasnet', 'SPHERE', 1) 
epoch = [1200, 800, 1000, 1100]
cltsize_lst = [1, 5, 10, 15, 20, 25, 30, 35]
SEED_LIST = [1, 2, 3]
batch_size = 8

for SEED in SEED_LIST:
    for cltsize in cltsize_lst:
        BASH_COMMAND_LIST.append(f"python train_toydata.py --dir_name 'log/train_synthetic/' " \
                                f"--tbd_dir 'log/tensorboard/train_synthetic' " \
                                f"--data_base_dir 'dataset/synthetic_data/cubesphere_1000' " \
                                f"--SVR " \
                                f"--network {model[0]} " \
                                f"--template_type {model[1]} " \
                                f"--nb_primitives {model[2]} " \
                                f"--bottleneck_size 1024 " \
                                f"--hidden_neurons 512 " \
                                f"--train_json_file 'cluster_image_aug_2by10_cltsize{cltsize:02}.json' " \
                                f"--test_json_file 'test_interp_1000.json' " \
                                f"--val_json_file 'val_interp_1000.json' " \
                                f"--batch_size {batch_size} " \
                                f"--batch_size_test 64 " \
                                f"--manual_seed {SEED} " \
                                f"--lrate 1e-3 " \
                                f"--workers 4 " \
                                f"--nepoch {epoch[0]} " \
                                f"--lr_decay_1 {epoch[1]} " \
                                f"--lr_decay_2 {epoch[2]} " \
                                f"--lr_decay_3 {epoch[3]} " \
                                f"--description 'cluster_image_aug_2by10_cltsize{cltsize:02}_seed{SEED}_BS{batch_size}_Epoch{epoch[0]}' ")
    # Train 
    for cltsize in cltsize_lst:
        BASH_COMMAND_LIST.append(f"python train_toydata.py --dir_name 'log/train_synthetic/' " \
                                f"--tbd_dir 'log/tensorboard/train_synthetic' " \
                                f"--data_base_dir 'dataset/synthetic_data/cubesphere_1000' " \
                                f"--SVR " \
                                f"--network {model[0]} " \
                                f"--template_type {model[1]} " \
                                f"--nb_primitives {model[2]} " \
                                f"--bottleneck_size 1024 " \
                                f"--hidden_neurons 512 " \
                                f"--train_json_file 'cluster_shape_aug_2by10_cltsize{cltsize:02}.json' " \
                                f"--test_json_file 'test_interp_1000.json' " \
                                f"--val_json_file 'val_interp_1000.json' " \
                                f"--batch_size {batch_size} " \
                                f"--batch_size_test 64 " \
                                f"--manual_seed {SEED} " \
                                f"--lrate 1e-3 " \
                                f"--workers 4 " \
                                f"--nepoch {epoch[0]} " \
                                f"--lr_decay_1 {epoch[1]} " \
                                f"--lr_decay_2 {epoch[2]} " \
                                f"--lr_decay_3 {epoch[3]} " \
                                f"--description 'cluster_shape_aug_2by10_cltsize{cltsize:02}_seed{SEED}_BS{batch_size}_Epoch{epoch[0]}' ")


dispatch_thread = DispatchThread("synthetic dataset training", 
                 BASH_COMMAND_LIST, logger, gpu_m_th=500, gpu_list=args.gpus, maxcheck=5)
# Start new Threads
dispatch_thread.start()
dispatch_thread.join()

import time
time.sleep(5)

logger.info("Exiting Main Thread")

