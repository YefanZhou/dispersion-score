import os
import torch
import sys
import logging
import time
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import statistics 
from os.path import join
#from model.model import EncoderDecoder

def get_logger(name, path, fname):
    """create a logger and return 
    """
    logger = logging.getLogger(name)
    file_log_handler = logging.FileHandler(os.path.join(path, fname))
    stderr_log_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(file_log_handler)
    logger.addHandler(stderr_log_handler)
    logger.setLevel("INFO")
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s","%Y-%m-%d %H:%M:%S")
    file_log_handler.setFormatter(formatter)
    stderr_log_handler.setFormatter(formatter)
    sys.stdout.flush()

    return logger

def CountFrequency(my_list): 
    """count the occurence in a list 
    """
    # Creating an empty dictionary  
    freq = {} 
    for item in my_list: 
        if (item in freq): 
            freq[item] += 1
        else: 
            freq[item] = 1

    freq = dict(sorted(freq.items()))
    return freq


def dic_to_array(dic):
    """convert dictionary to numpy array for saving
    """
    dim = len(dic)
    array = np.zeros(dim)
    for idx, key in enumerate(dic):
        array[idx] = dic[key]

    return array


def array_to_dic(array, keys=None):
    """
    """
    res = {}
    if keys:
        for i in range(array.shape[0]):
            res.update({keys[i]: array[i]})
    else:
        for i in range(array.shape[0]):
            res.update({str(i): array[i]})

    return res


def mean_std(test_list):
    """calculate mean and std
    """
    mean = sum(test_list) / len(test_list) 
    variance = sum([((x - mean) ** 2) for x in test_list]) / len(test_list) 
    std = variance ** 0.5

    return mean, std


def verbose_dic(dictionary):
    """
    """
    verbose = ""
    for key in dictionary:
        verbose += f"{key}: {dictionary[key] :2f}  "
    
    return verbose



def dic_to_box(dic):
    """
    """
    boxes = [
        {
            'label' : "Male height",
            'whislo': dic['min'],    # Bottom whisker position
            'q1'    : dic['lower_quat'],    # First quartile (25th percentile)
            'med'   : dic['median'],    # Median         (50th percentile)
            'q3'    : dic['up_quat'],    # Third quartile (75th percentile)
            'whishi': dic['max'],    # Top whisker position
            'fliers': []        # Outliers
        }
    ]

    return boxes


def list_to_indexdic(list):
    unique_lookup = {}
    lookup = {}
    unique_idx = 0
    for item in list:
        if item not in unique_lookup:
            unique_lookup.update({item: unique_idx})
            unique_idx += 1
    
    for idx, item in enumerate(list):
        lookup.update({idx: unique_lookup[item]})

    return lookup



def build_graph():
    pass


def label_to_colr_mapper(label_list, colormap):
    """
    """
    color_cat_dic = list_to_indexdic(label_list)
    low, *_, high = sorted(color_cat_dic.values())
    norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=colormap) 

    return mapper, color_cat_dic



def model_eval(exp_opts, opt, data, batch_size, logger):
    """
    Generate recon point cloud based on trained model  

    Parameters:
    --------------------
    :params: exp_opts
    :params: opt
    :params: data
    """
    if exp_opts.network == 'atlasnet':
        network = EncoderDecoder(exp_opts)
    else:
        raise NotImplementedError

    network.load_state_dict(torch.load(join(opt.trained_exp_dir, "network.pth"), map_location='cuda'))
    logger.info(f"Network weights loaded from  {opt.trained_exp_dir}!")

    network.eval()
    pred_list = []
    with torch.set_grad_enabled(False): 
        for input_data in tqdm.tqdm(data.split(batch_size), total=data.shape[0]//batch_size):
            pred = network(input_data.to(opt.device))
            if exp_opts.network == 'atlasnet':
                pred = pred.transpose(2, 3).contiguous()
                pred = pred.view(pred.shape[0], -1, 3)
            pred_list.append(pred.cpu())
    pred_points = torch.cat(pred_list, dim=0)

    return pred_points

