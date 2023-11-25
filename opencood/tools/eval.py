# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os, glob
import time
from tqdm import tqdm

import torch
import open3d as o3d
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils
import matplotlib.pyplot as plt


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy_test file')
    parser.add_argument('--save_result', action='store_true',
                        help='whether to save prediction and gt result'
                             'in .pth file')
    parser.add_argument('--global_sort_detections', action='store_true',
                        help='whether to globally sort detections by confidence score.'
                             'If set to True, it is the mainstream AP computing method,'
                             'but would increase the tolerance for FP (False Positives).')
    opt = parser.parse_args()
    return opt


def main(saved_path):
    filenames = sorted(glob.glob(os.path.join(saved_path, '*.pth')))

    # Create the dictionary for evaluation.
    # also store the confidence score for each prediction
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    for f in tqdm(filenames):
        data = torch.load(f)
        pred_box_tensor = data['pred']
        pred_score = data['score']
        gt_box_tensor = data['gt']

        eval_utils.caluclate_tp_fp(pred_box_tensor,
                                   pred_score,
                                   gt_box_tensor,
                                   result_stat,
                                   0.3)
        eval_utils.caluclate_tp_fp(pred_box_tensor,
                                   pred_score,
                                   gt_box_tensor,
                                   result_stat,
                                   0.5)
        eval_utils.caluclate_tp_fp(pred_box_tensor,
                                   pred_score,
                                   gt_box_tensor,
                                   result_stat,
                                   0.7)


    eval_utils.eval_final_results(result_stat,
                                  saved_path,
                                  False)


if __name__ == '__main__':
    main("/home/projects/OpenCOOD/ckpt/voxelnet_attentive_fusion/voxelnet_attentive_fusion/result")
