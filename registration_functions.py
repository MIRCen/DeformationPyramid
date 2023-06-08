
import config as cfg

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
BCE = nn.BCELoss()
import open3d as o3d
import logging
import os
from easydict import EasyDict as edict
import numpy as np
import torch.optim as optim
from model.nets import Deformation_Pyramid
from model.loss import compute_truncated_chamfer_distance
config = edict(cfg.config)
if config.gpu_mode:
    config.device = torch.cuda.current_device()
else:
    config.device = torch.device('cpu')


#src_pc_file = src_pc_file.farthest_point_down_sample(num_samples=85000)##uniform_down_sample(every_k_points=5)


def apply_shape_transfer(NDP, src_pcd,min_level=0, max_level=8):
    #NDP.gradient_setup(optimized_level=-1)
    src_pcd_device = torch.from_numpy(src_pcd).to(config.device)
    warped_src, data = NDP.warp(src_pcd_device,min_level=min_level,max_level=max_level)
    warped_src = warped_src.detach().cpu().numpy()
    return warped_src


def shape_transfer_2D_2D(tgt_pcd, src_pcd, m=config.m, trunc=config.trunc, motion_type=config.motion_type):
    src_pcd, tgt_pcd = map(lambda x: torch.from_numpy(x).to(config.device), [src_pcd, tgt_pcd])
    NDP = Deformation_Pyramid(depth=config.depth,
                              width=config.width,
                              device=config.device,
                              k0=config.k0,
                              m=m,
                              nonrigidity_est=config.w_reg > 0,
                              rotation_format=config.rotation_format,
                              motion=motion_type)
    s_sample = src_pcd
    t_sample = tgt_pcd

    loss = compute_truncated_chamfer_distance(s_sample[None], t_sample[None], trunc=trunc)
    print(f"Initial loss : {loss}")

    for level in range(NDP.n_hierarchy):
        """freeze non-optimized level"""
        NDP.gradient_setup(optimized_level=level)
        optimizer = optim.Adam(NDP.pyramid[level].parameters(), lr=config.lr)
        break_counter = 0
        loss_prev = 1e+6

        """optimize current level"""
        for iter in range(config.iters):

            s_sample_warped, data = NDP.warp(s_sample, max_level=level, min_level=level)
            loss = compute_truncated_chamfer_distance(s_sample_warped[None], t_sample[None], trunc=trunc)

            if level > 0 and config.w_reg > 0:
                nonrigidity = data[level][1]
                target = torch.zeros_like(nonrigidity)
                reg_loss = BCE(nonrigidity, target)
                loss = loss + config.w_reg * reg_loss

            # early stop
            if loss.item() < 1e-4:
                break
            if abs(loss_prev - loss.item()) < loss_prev * config.break_threshold_ratio:
                break_counter += 1
            if break_counter >= config.max_break_count:
                break
            loss_prev = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # use warped points for next level
        s_sample = s_sample_warped.detach()
        print(f"Level : {level}, Loss : {loss},  Iter : {iter}, break_counter {break_counter}")
    return NDP, loss


def chamfer_correspondence_mouse_2D_rigid(tgt_pc_list, src_pc_list):
    len_tgt_pc_list = len(tgt_pc_list)
    len_src_pc_list = len(src_pc_list)
    loss_matrix = np.zeros((len_tgt_pc_list, len_src_pc_list))
    for i in range(len_tgt_pc_list):
        for j in range(len_src_pc_list):
            _, loss_rigid = shape_transfer_2D_2D(tgt_pc_list[i], src_pc_list[j], m=1)
            loss_matrix[i, j] = loss_rigid

    return loss_matrix


def shape_transfer_multiple_slices_2D_2D(tgt_pc_list, src_pc_list, m=9, trunc = 100):
    len_tgt_pc_list = len(tgt_pc_list)
    NDP_list = []
    loss_list = []
    for i in range(len_tgt_pc_list):
        NDP, loss = shape_transfer_2D_2D(tgt_pc_list[i], src_pc_list[i], m=m, trunc = trunc)
        NDP_list.append(NDP)
        loss_list.append(loss)
    return NDP_list, loss_list

def slices_alignment_2D_2D(src_pc_list, m=1, motion_type ="SE3"):
    len_src_pc_list= len(src_pc_list)
    NDP_list=[]
    loss_list=[]
    aligned_src_pc_list=[ src_pc_list[0]]


    for i in range(1,len_src_pc_list):
        NDP, loss = shape_transfer_2D_2D(src_pc_list[i-1], src_pc_list[i], m=m, motion_type =motion_type)
        aligned_src=apply_shape_transfer(NDP, src_pc_list[i], max_level=0)
        NDP_list.append(NDP)
        loss_list.append(loss)
        aligned_src_pc_list.append(aligned_src)

    return aligned_src_pc_list, loss_list



def save_model(model,  save_dir,  name=None):
    """
    state = {
        'state_dict': model.state_dict(),
    }
    if name is None:
        filename = os.path.join(save_dir, f'model_{epoch}.pth')
    else:
    """
    filename = os.path.join(save_dir, f'model_{name}.pth')
    torch.save(model, filename, _use_new_zipfile_serialization=False)

def load(save_dir, model):
        print ("loading trained", save_dir)
        if os.path.isfile(save_dir):
            state = torch.load(save_dir)
            model.load_state_dict(state['state_dict'])
        else:
            raise ValueError(f"=> no model found at '{save_dir}'")


