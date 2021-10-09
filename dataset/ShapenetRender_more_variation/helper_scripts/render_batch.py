import os
import sys
import time
from joblib import Parallel, delayed
import argparse

#
parser = argparse.ArgumentParser()
parser.add_argument('--model_root_dir', type=str, default="/home/yirus/Datasets/PartSeg/train_test")
parser.add_argument('--blender_location', type=str, default="/usr/local/blender-2.79-linux-glibc219-x86_64/blender")
parser.add_argument('--num_thread', type=int, default=10, help='1/3 of the CPU number')
parser.add_argument('--num_views', type=int, default=9, help='number of views to be rendered')
FLAGS = parser.parse_args()

model_root_dir = FLAGS.model_root_dir
# filelist_dir = FLAGS.filelist_dir

cat_ids = {
    "Airplane": "02691156",
    "Bag": "02773838",
    "Cap": "02954340",
    "Car": "02958343",
    "Chair": "03001627",
    "Earphone": "03261776",
    "Guitar": "03467517",
    "Knife": "03624134",
    "Lamp": "03636649",
    "Laptop": "03642806",
    "Motorbike": "03790512",
    "Mug": "03797390",
    "Pistol": "03948459",
    "Rocket": "04099429",
    "Skateboard": "04225987",
    "Table": "04379243",
}

id_to_cat = {
    "02691156": "Airplane",
    "02773838": "Bag",
    "02954340": "Cap",
    "02958343": "Car",
    "03001627": "Chair",
    "03261776": "Earphone",
    "03467517": "Guitar",
    "03624134": "Knife",
    "03636649": "Lamp",
    "03642806": "Laptop",
    "03790512": "Motorbike",
    "03797390": "Mug",
    "03948459": "Pistol",
    "04099429": "Rocket",
    "04225987": "Skateboard",
    "04379243": "Table",
}



def gen_obj(model_root_dir, cat_id, obj_id, num_views):
    # if os.path.exists(os.path.join(model_root_dir, "rendering_metadata.txt")):
    #     print("Exist!!!, skip %s %s" % (cat_id, obj_id))
    # else:
    print("Start %s %s" % (cat_id, obj_id))
    obj_path = os.path.join(model_root_dir, cat_id, obj_id)
    os.system(FLAGS.blender_location + ' --background --python render_blender.py -- --num_views %d --obj_dir %s' % (num_views, obj_path))

    print("Finished %s %s"%(cat_id, obj_id))
#

for cat_id in os.listdir(FLAGS.model_root_dir):
    print("........................................................................................................")
    print("Processing {} ....................................................................................".format(cat_id))
    obj_list = os.listdir(os.path.join(FLAGS.model_root_dir, cat_id))

    ml, cl, ol = [], [], []
    # print(cat_id, id_to_cat[cat_id], len(obj_list))
    for obj_id in obj_list:
        ml.append(model_root_dir)
        cl.append(cat_id)
        ol.append(obj_id)
    # print(ml[-1], cl[-1], ol[-1])
    # break
    # gen_obj(model_root_dir, cat_id, obj_id, FLAGS.num_views)
    #     print(obj_id)
    # with Parallel(n_jobs=5) as parallel:
    #     parallel(delayed(gen_obj)(model_root_dir, cat_id, obj_id, FLAGS.num_views))
    with Parallel(n_jobs=5) as parallel:
        parallel(delayed(gen_obj)(model_root_dir, cat_id, obj_id, FLAGS.num_views) for
                 model_root_dir, cat_id, obj_id in
                 zip(ml, cl, ol))
    # exit()
    print("Finished %s" % cat_id)