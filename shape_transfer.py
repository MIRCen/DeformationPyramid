import torch
import torch.nn as nn

BCE = nn.BCELoss()
import open3d as o3d
import logging
import os
import time
from pathlib import Path
import torch.optim as optim
from easydict import EasyDict as edict
from model.nets import Deformation_Pyramid
import torch
from utils.benchmark_utils import setup_seed
import numpy as np

from visualisation import print_points_cloud_information, visualize_registration, visualize_label, visualize_deformation
from data_preparation.data_preparation import read_file_info, inverse_data_normalisation, get_list_ROI
from data_preparation.dataloaders import Mouse_pc, get_labels
from registration_functions import shape_transfer_2D_2D, shape_transfer_multiple_slices_2D_2D, apply_shape_transfer, \
    save_model, slices_alignment_2D_2D
import config as cfg

setup_seed(0)

if __name__ == "__main__":
    config = edict(cfg.config)
    if config.gpu_mode:
        config.device = torch.cuda.current_device()
    else:
        config.device = torch.device('cpu')

    # define path
    data_name = "10_TI2202_mouse-NeuN_G-Liot"
    project_name = "10_brain_region_segmentation"
    tgt_file = "reference"
    src_file = "source"
    results_file = "results"
    data_path = Path(__file__)

    data_path = Path(__file__).parent.joinpath("datasets", data_name, project_name)
    data_tgt_roi_path = data_path.joinpath(tgt_file, "57_ROI_img")
    data_tgt_params_path = data_path.joinpath(tgt_file, "70_paramap")
    data_tgt_images = data_path.joinpath(tgt_file, "20_gis")
    data_tgt_info_slices = data_path.joinpath(tgt_file, "info_mouse_slices.ods")

    data_src_roi_path = data_path.joinpath(src_file, "57_ROI_img")
    data_src_params_path = data_path.joinpath(src_file, "70_paramap")
    data_src_images = data_path.joinpath(src_file, "20_gis")
    data_src_info_slices = data_path.joinpath(src_file, "info_mouse_slices.ods")

    data_results_path = data_path.joinpath(results_file)

    # load data
    tgt_slices_numeros, tgt_rotations, tgt_image_dim = read_file_info(data_tgt_info_slices)
    src_slices_numeros, src_rotations, src_image_dim = read_file_info(data_src_info_slices)

    # Construct Mouse class
    Mouse_tgt = Mouse_pc(data_tgt_params_path, data_ROI_path=data_tgt_roi_path,
                         image_sizes=tgt_image_dim)
    Mouse_src = Mouse_pc(data_src_params_path, data_ROI_path=data_src_roi_path,
                         image_sizes=src_image_dim, rotation=src_rotations)

    # Choose slices
    index_start = 0
    index_end = 10
    tgt_slice_interval = Mouse_tgt.get_slices_pc_interval(index_start, index_end)
    src_slice_interval = Mouse_src.get_slices_pc_interval(index_start, index_end)

    """
    #Compute shape transfer
    NDP_list,loss_list=shape_transfer_multiple_slices_2D_2D(tgt_slice_interval, src_slice_interval, m=9,trunc = 0.0002)

    #Save models
    for i in range(index_start, index_end):
        save_model(NDP_list[i-index_start],data_results_path,name=i)
    """
    # load models
    NDP_list = []
    for i in range(index_start, index_end):
        model = torch.load(os.path.join(data_results_path, f'model_{i}.pth'))
        NDP_list.append(model)
    # Apply NDP, get labels
    list_labels = []
    list_warped_src = []

    for i in range(index_start, index_end):
        warped_src_b = apply_shape_transfer(NDP_list[i - index_start], Mouse_src.get_slice_pc(i))
        labeled_nodes = get_labels(Mouse_tgt, warped_src_b, i)
        list_labels.append(labeled_nodes)
        list_warped_src.append(warped_src_b)
        # grid = visualize_deformation(NDP_list[i])

    # display 3D segmentation
    aligned_src_pc_list, _ = slices_alignment_2D_2D(src_slice_interval, m=1, motion_type="SE3")

    for index in range(index_start, index_end, 1):
        slice = aligned_src_pc_list[index]
        slice[:, 2] += index * 0.1

    c_aligned_src_pc_list = np.concatenate(aligned_src_pc_list, axis=0)
    c_labels = np.concatenate(list_labels, axis=0)
    visualize_label(c_aligned_src_pc_list, c_labels, region=1)

    c_slices = Mouse_src.merge_point_cloud(index_start, index_end)
    c_labels = np.concatenate(list_labels, axis=0)
    visualize_label(c_slices, c_labels, region=1)

    # Display result by slices
    for i in range(index_end):
        visualize_registration(Mouse_tgt.get_slice_pc(i), Mouse_src.get_slice_pc(i), list_warped_src[i])
        # visualize_label(Mouse_src.get_slice_pc(i), list_labels[i])
