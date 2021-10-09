import os
import argparse
from os.path import join

parser = argparse.ArgumentParser()
parser.add_argument("--model_root_dir", type=str, default="/home/ubuntu/ShapeNetV1/ShapeNetCore.v1", help="path to ShapeNet mesh model")
parser.add_argument("--blender_path", type=str, default="/usr/local/blender-2.79b-linux-glibc219-x86_64/blender", help="blender executable path")
parser.add_argument("--render_root_dir", type=str, default="../data/ShapeNetV1RenderingAnglelimit", help="saving base dir")

opt = parser.parse_args()

COMMAND = []
y_rot_angle_lst = [0, 15, 30, 45, 60, 75, 90] #90, 30, 120, 60, 150

for y_rot_angle in y_rot_angle_lst:
    save_dir = join(opt.render_root_dir, f'ShapeNetV1RenderingSampleYaw{y_rot_angle:03}')
    os.makedirs(save_dir, exist_ok=True)
    COMMAND = f"python -u render_batch_anglecontrol.py --model_root_dir {opt.model_root_dir} " \
                    f"--render_root_dir {save_dir} " \
                    f"--filelist_dir './filelists' " \
                    f"--blender_location {opt.blender_path} " \
                    f"--num_thread 5 " \
                    f"--shapenetversion 'v1' " \
                    f"--y_rot_range {y_rot_angle} " \
                    f"--views 1 "

    os.system(COMMAND)