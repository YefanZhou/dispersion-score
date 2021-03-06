import argparse
import auxiliary.my_utils as my_utils
import os
import datetime
import json
import sys
import logging
from termcolor import colored
from easydict import EasyDict
import tensorboardX
import yaml
from os.path import exists, join


def parser():
    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument("--no_learning", action="store_true", default=False, help="only train the encoder")
    parser.add_argument("--train_only_encoder", action="store_true", help="only train the encoder")
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--batch_size_test', type=int, default=32, help='input batch size')
    parser.add_argument('--parallel', action="store_true", help='use data parallel')
    parser.add_argument('--nepoch', type=int, default=150, help='number of epochs to train for')
    parser.add_argument('--start_epoch', type=int, default=0, help='number of epochs to train for')
    parser.add_argument("--manual_seed", type=int, default=1, help="if seed > 0, then fixed, if < 0 then random")
    parser.add_argument('--lrate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_1', type=int, default=10000000, help='learning rate decay 1')
    parser.add_argument('--lr_decay_2', type=int, default=10000000, help='learning rate decay 2')
    parser.add_argument('--lr_decay_3', type=int, default=10000000, help='learning rate decay 3')
    parser.add_argument('--lr_decay_4', type=int, default=10000000, help='learning rate decay 4')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for toydataset')
    parser.add_argument("--run_single_eval", action="store_true", help="evaluate a trained network")
    parser.add_argument("--demo", action="store_true", help="run demo autoencoder or single-view")

    # Data
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
    parser.add_argument("--views_search", action="store_true", default=False, help="increment views to train")
    parser.add_argument("--random_view_select", action="store_true", default=False, help="random select one view every epoch in 23 views in training")
    parser.add_argument('--anglelimit_search', action='store_true', default=False, help='use limited yaw rendering')

    parser.add_argument("--mode", type=str, default="viewer", choices=['viewer', 'object'])
    parser.add_argument("--nviews_train", type=int, default=1, help='num of view per shape for training')
    parser.add_argument("--nviews_test", type=int, default=1, help='num of view per shape for test')
    parser.add_argument('--normalization', type=str, default="UnitBall",
                        choices=['UnitBall', 'BoundingBox', 'Identity'])
    parser.add_argument("--shapenet13", action="store_true", help="Load 13 usual shapenet categories")
    parser.add_argument("--SVR", action="store_true", help="Single_view Reconstruction")
    parser.add_argument("--sample", action="store_false", help="Sample the input pointclouds")
    parser.add_argument('--class_choice', nargs='+', default=["table", "car"], type=str)
    parser.add_argument('--number_points', type=int, default=2500, help='Number of point sampled on the object during training, and generated by atlasnet')
    parser.add_argument('--number_points_eval', type=int, default=2500,
                        help='Number of points generated by atlasnet (rounded to the nearest squared number) ')

    parser.add_argument("--img_aug", action="store_true", help="apply image augmentation like random crop")
    parser.add_argument("--img_aug_type", type=str, default='rgb', choices=['grayscale', 'rgb', 'binary', 'color_aug', 'color_aug_random', 'autoaugment', 'geometricaugment', 'geometricaugmentMag', 'coloraugment'])
    parser.add_argument("--autoaug_type", type=str, default='ImageNet', choices=['ImageNet', 'CIFAR10', 'Seq', 'RGB', 'SVHN', 'None', 'SeqBlackBG'])
    parser.add_argument("--color_aug_factor", nargs='+', type=float, default=[1.0, 1.0, 1.0, 1.0], help='brightness, contrast, saturation, hue')
    parser.add_argument("--test_augment", action="store_true", default=False, help='test augment')
    parser.add_argument("--mag_idx", default=0, type=int, help="magnitude level index")
    parser.add_argument("--magnitude", default=0, type=int, help="magnitude for coloraug")
    parser.add_argument("--n_op", default=0, type=int, help="number of operations")
    parser.add_argument("--prob", default=1, type=float, help="prob")
    
    parser.add_argument("--random_rotation", action="store_true", help="apply data augmentation : random rotation")
    parser.add_argument("--data_augmentation_axis_rotation", action="store_true",
                        help="apply data augmentation : axial rotation ")
    parser.add_argument("--data_augmentation_random_flips", action="store_true",
                        help="apply data augmentation : random flips")
    parser.add_argument("--random_translation", action="store_true",
                        help="apply data augmentation :  random translation ")
    parser.add_argument("--anisotropic_scaling", action="store_true",
                        help="apply data augmentation : anisotropic scaling")

    # Save dirs and reload
    parser.add_argument('--id', type=str, default="0", help='training name')
    parser.add_argument('--description', type=str, default="", help='descript this training')
    parser.add_argument('--tbd_dir', type=str, default="log/tensorboard", help='name of the tensorboard folder.')
    parser.add_argument('--train_endsignal', type=str, default="train_end.txt", help='ending signal for scheduler.')
    
    parser.add_argument('--dir_name', type=str, default="log/oneExp", help='name of the log folder.')
    parser.add_argument('--save_pred', action="store_true", help="save prediction points")
    parser.add_argument('--demo_input_path', type=str, default="./doc/pictures/plane_input_demo.png", help='dirname')
    parser.add_argument('--reload_options_path', type=str, default=None, help='options path')
    parser.add_argument('--reload_decoder_path', type=str, default="", help='dirname')
    parser.add_argument('--reload_model_path', type=str, default='', help='optional reload model path')
    parser.add_argument('--reload_optimizer_path', type=str, default='', help='optional reload model path')

    # Network
    parser.add_argument("--encoder", type=str, default="resnet18", help="[resnet18, resnet50]")
    parser.add_argument("--network", type=str, default="atlasnet", help="[psgn, atlasnet, foldnet, oraclenn, clustering]")
    parser.add_argument('--num_layers', type=int, default=2, help='number of hidden MLP Layer')
    parser.add_argument('--hidden_neurons', type=int, default=512, help='number of neurons in each hidden layer')
    parser.add_argument('--loop_per_epoch', type=int, default=1, help='number of data loop per epoch')
    parser.add_argument('--nb_primitives', type=int, default=25, help='number of primitives')
    parser.add_argument('--template_type', type=str, default="SPHERE", choices=["SPHERE", "SQUARE", 'GAUSSIAN'],
                        help='dim_out_patch')
    parser.add_argument('--multi_gpu', nargs='+', type=int, default=[0], help='Use multiple gpus')
    parser.add_argument("--remove_all_batchNorms", action="store_true", help="Replace all batchnorms by identity")
    parser.add_argument('--bottleneck_size', type=int, default=1024, help='dim_out_patch')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=["relu", "sigmoid", "softplus", "logsigmoid", "softsign", "tanh"], help='dim_out_patch')
    parser.add_argument('--no_compile_chamfer', action="store_true", help="compile c++ version chamfer distance")
    parser.add_argument('--dismat_path', type=str, 
                    default="log/benchmark/20210202_2351_viewer_oraclenn_seed1/viewer_oraclenn_trainnv1.npy", 
                    help="path to distance matrix for Oracle NN or Clustering")
    parser.add_argument('--train_json_file', type=str, default='cluster_image_aug_2by10_cltsize01.json', help='toy dataset json file for train')
    parser.add_argument('--test_json_file', type=str, default='test_interp_1000.json', help='toy dataset json file for test')
    parser.add_argument('--val_json_file', type=str, default='val_interp_1000.json', help='toy dataset json file for val')
    
    parser.add_argument('--data_base_dir', type=str, default="dataset/synthetic_data/cubesphere_1000", help='data base folder for synthetic dataset')
    parser.add_argument('--rendering_root_dir', type=str, default="",
                                                            help='')
    parser.add_argument('--yawrange', type=int, default=0, help='')
    

    # Loss
    parser.add_argument("--no_metro", action="store_false", help="Compute metro distance")

    opt = parser.parse_args()

    opt.date = str(datetime.datetime.now())
    now = datetime.datetime.now()
    opt = EasyDict(opt.__dict__)
    logger = logging.getLogger()
    if not opt.run_single_eval:
    # Set-up output directories
        net_desc = '{}_{}'.format(now.strftime("%Y%m%d_%H%M"), '_'.join(opt.description.split()))
        opt.dir_name = os.path.join(opt.dir_name, net_desc)
        opt.tbd_dir = os.path.join(opt.tbd_dir, net_desc)
        my_utils.check_exist_or_mkdirs(opt.dir_name)
        my_utils.check_exist_or_mkdirs(opt.tbd_dir)
        opt.description = net_desc

    file_log_handler = logging.FileHandler(os.path.join(opt.dir_name, "Train.log"))
    logger.addHandler(file_log_handler)
    stderr_log_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stderr_log_handler)
    logger.setLevel("INFO")
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s","%Y-%m-%d %H:%M:%S")
    file_log_handler.setFormatter(formatter)
    stderr_log_handler.setFormatter(formatter)
    sys.stdout.flush()

    # If running a demo, check if input is an image or a pointcloud
    if opt.demo:
        ext = opt.demo_input_path.split('.')[-1]
        if ext == "ply" or ext == "npy" or ext == "obj":
            opt.SVR = False
        elif ext == "png":
            opt.SVR = True

    # if opt.demo or opt.run_single_eval:
    #     if not exists("./training/trained_models/atlasnet_singleview_25_squares/network.pth"):
    #         logger.info("Dowload Trained Models.")
    #         os.system("chmod +x training/download_trained_models.sh")
    #         os.system("./training/download_trained_models.sh")

    #     if opt.reload_model_path == "" and opt.SVR:
    #         opt.dir_name = "./training/trained_models/atlasnet_singleview_1_sphere"
    #     elif opt.reload_model_path == "" and not opt.SVR:
    #         opt.dir_name = "./training/trained_models/atlasnet_autoencoder_1_sphere"

    # if exists(join(opt.dir_name, "options.json")):
    #     # Reload parameters from options.txt if it exists
    #     with open(join(opt.dir_name, "options.json"), 'r') as f:
    #         my_opt_dict = json.load(f)
    #     my_opt_dict.pop("run_single_eval")
    #     my_opt_dict.pop("no_metro")
    #     my_opt_dict.pop("train_only_encoder")
    #     my_opt_dict.pop("no_learning")
    #     my_opt_dict.pop("demo")
    #     my_opt_dict.pop("demo_input_path")
    #     my_opt_dict.pop("dir_name")
    #     for key in my_opt_dict.keys():
    #         opt[key] = my_opt_dict[key]
    #     if not opt.demo:
    #         logger.info("Modifying input arguments to match network in dirname")
    #         logger.info("PARAMETER: ")
    #         for a in my_opt_dict:
    #             logger.info(
    #                 "         "
    #                 + a
    #                 + " : "
    #                 + str(my_opt_dict[a])
    #             )

    # if opt.reload_options_path is not None and exists(opt.reload_options_path):
    #     # Reload parameters from options.txt if it exists
    #     logger.info(f"Reload options from {opt.reload_options_path}")
    #     with open(opt.reload_options_path, 'r') as f:
    #         my_opt_dict = json.load(f)
    #     my_opt_dict.pop("reload_model_path")
    #     my_opt_dict.pop("run_single_eval")
    #     my_opt_dict.pop("log_path")
    #     my_opt_dict.pop("training_media_path")
    #     my_opt_dict.pop("description")
    #     my_opt_dict.pop("tbd_dir")
    #     my_opt_dict.pop("no_metro")
    #     my_opt_dict.pop("train_only_encoder")
    #     my_opt_dict.pop("no_learning")
    #     my_opt_dict.pop("demo")
    #     my_opt_dict.pop("demo_input_path")
    #     my_opt_dict.pop("dir_name")
    #     for key in my_opt_dict.keys():
    #         opt[key] = my_opt_dict[key]
    #     if not opt.demo:
    #         logger.info("Combind Options after merging")
    #         logger.info("PARAMETER: ")
    #         for a in opt:
    #             logger.info(
    #                 "         "
    #                 + a
    #                 + " : "
    #                 + str(opt[a])
    #             )

    if not opt.run_single_eval:
        with open(os.path.join(opt.dir_name, 'opts.yaml'), 'w') as file:
            yaml.dump(dict(opt), file, default_flow_style=False, sort_keys=False)

    # Hard code dimension of the template.
    dim_template_dict = {
        "SQUARE": 2,
        "SPHERE": 3,
    }

    opt.dim_template = dim_template_dict[opt.template_type]

    opt.logger = logger
    if not opt.run_single_eval:
        opt.tbwriter = tensorboardX.SummaryWriter(opt.tbd_dir)
    else:
        opt.tbwriter = None

    if opt.run_single_eval and opt.save_pred:
        opt.results_dir = os.path.join(opt.dir_name, 'results')
        my_utils.check_exist_or_mkdirs(opt.results_dir)


    if len(opt.multi_gpu) > 1 and opt.parallel:
        opt.logger.info(f"DATA Parallel Enabled in GPU {opt.multi_gpu} !")

    return opt
