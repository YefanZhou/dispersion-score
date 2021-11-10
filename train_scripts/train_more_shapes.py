import os
import sys
import signal
sys.path.append('../')
from training.gputracker import get_logger
###################################################################################################
#############################ShapeNet 13 OC VC ######################################
logger = get_logger('log', 'schedule_more_shapes.log')

MODE_OPTIONS=['viewer', 'object']
MODEL_OPTIONS=[('atlasnet', 'SPHERE', 1)]
DECAY_OPTIONS=[(120, 90, 110, 115)]
TRAIN_VIEW_OPTIONS=[1]
TRAIN_VIEWS=1
TEST_VIEWS=1
SEED_LIST = [1, 2, 3]
BASH_COMMAND_LIST=[]
## Train
for SEED in SEED_LIST:
    for decay_idx, decay in enumerate(DECAY_OPTIONS):
        for mode_idx, mode in enumerate(MODE_OPTIONS):
            for model_idx, model in enumerate(MODEL_OPTIONS):
                BASH_COMMAND_LIST.append(f"python train.py --shapenet13 --SVR " \
                                        f"--dir_name 'log/more_shapes' " \
                                        f"--network {model[0]} " \
                                        f"--template_type {model[1]} " \
                                        f"--nb_primitives {model[2]} " \
                                        f"--batch_size 64 " \
                                        f"--batch_size_test 64 " \
                                        f"--workers 4 " \
                                        f"--lrate 1e-3 " \
                                        f"--tbd_dir 'log/tensorboard/more_shapes' " \
                                        f"--nepoch {decay[0]} " \
                                        f"--lr_decay_1 {decay[1]} " \
                                        f"--lr_decay_2 {decay[2]} " \
                                        f"--lr_decay_3 {decay[3]} " \
                                        f"--manual_seed {SEED} " \
                                        f"--views_search " \
                                        f"--mode {mode} " \
                                        f"--nviews_train {TRAIN_VIEWS} " \
                                        f"--nviews_test {TEST_VIEWS} " \
                                        f"--description 'more_shapes_{mode}_{model[0]}{model[1]}{model[2]}_seed{SEED}'")

for command in BASH_COMMAND_LIST:
    logger.info(f"Launching Experiments: {command}")
    os.system(command)

    # sleep 5 seconds 
    code = os.system('sleep 5')

    if code == signal.SIGINT:
        logger.info('Keyboard Interpret')
        break