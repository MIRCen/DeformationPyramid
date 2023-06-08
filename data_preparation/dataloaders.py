import config as cfg
import os
import PIL.Image as Image
import numpy as np
from data_preparation.data_preparation import get_list_ROI, read_aims_features
import torch
import torch.nn as nn

BCE = nn.BCELoss()
import open3d as o3d
import logging
import os
from easydict import EasyDict as edict
import numpy as np
from data_preparation.data_preparation import read_file_info, inverse_data_normalisation, get_list_ROI

config = edict(cfg.config)
if config.gpu_mode:
    config.device = torch.cuda.current_device()
else:
    config.device = torch.device('cpu')

from visualisation import print_points_cloud_information, visualize_registration, visualize_label


# src_pc_file = src_pc_file.farthest_point_down_sample(num_samples=85000)##uniform_down_sample(every_k_points=5)


class Mouse_pc():
    def __init__(self, data_params_path, data_ROI_path=None, image_sizes=None, rotation=None, selected_features=None):

        self.data_params_path = data_params_path
        self.data_ROI_path = data_ROI_path
        self.image_sizes = image_sizes
        self.params_urls = sorted([os.path.join(self.data_params_path, f) for f in os.listdir(self.data_params_path) if
                                   os.path.isfile(os.path.join(self.data_params_path, f))])
        print(f"Urls of loaded image parameters : {self.params_urls}")
        self.list_features = read_aims_features(self.params_urls)
        self.number_slices = len(self.list_features)
        if (self.data_ROI_path is not None) and (self.image_sizes is not None) and (
                len(self.image_sizes) == len(os.listdir(self.data_ROI_path)) == len(self.params_urls)):
            self.ROI_urls = sorted([os.path.join(self.data_ROI_path, f) for f in os.listdir(self.data_ROI_path) if
                                    os.path.isfile(os.path.join(self.data_ROI_path, f))])
            self.ROI_images = [np.array(Image.open(f)) for f in self.ROI_urls]
            print(f"Urls of loaded ROI images : {self.ROI_urls}")
        else:
            self.data_ROI_path = None
            print("No ROI available.")
        self.data_list = []
        self.label_list = []
        self.center_list = []

        for i, features in enumerate(self.list_features):
            if self.data_ROI_path is not None:
                raw_x, raw_y = np.asarray(features[:, 0], dtype=int), np.asarray(features[:, 1], dtype=int)
                labeled_nodes_original = get_list_ROI(raw_x, raw_y, self.ROI_images[i].T,
                                                      image_size=self.image_sizes[i])
                # print(np.unique(labeled_nodes))
                labeled_nodes = [label % 80 for label in labeled_nodes_original]
            if rotation is not None:
                [features[:, 0], features[:, 1]] = rotation_features(features, rotation[i])
            features, src_middle = self.normalize_coor_slice_pc(i)

            self.data_list.append(features)
            self.label_list.append(labeled_nodes)
            self.center_list.append(src_middle)

    def get_slice_pc(self, index):
        return self.list_features[index]

    def get_slices_pc_interval(self, index_start, index_end):
        return [self.list_features[index] for index in range(index_start, index_end, 1)]

    def merge_point_cloud(self, index_start, index_end):
        slices = self.get_slices_pc_interval(index_start, index_end)
        for index in range(index_start, index_end, 1):
            slice = slices[index]
            slice[:, 2] += index * 0.1
        return np.concatenate(slices, axis=0)

    def normalize_coor_slice_pc(self, index):
        features = self.list_features[index]
        features[:, 1] = - features[:, 1]
        src_percentile_99 = np.percentile(features[:, 0:2], 99, axis=0)
        src_percentile_01 = np.percentile(features[:, 0:2], 1, axis=0)
        src_middle = (src_percentile_99 + src_percentile_01) / 2
        features[:, 0:2] = (features[:, 0:2] - src_middle) / [20000, 15000]
        return features, src_middle

    def invert_normalisation_slice(self, index, middle=None):
        features = self.get_slice_pc(index)
        features[:, 0:2] = features[:, 0:2] * [20000, 15000] + middle
        features[:, 1] = -features[:, 1]
        return features


def rotation_features(features, rotation):
    theta = np.radians(rotation)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return np.dot(R, [features[:, 0], features[:, 1]])


def get_labels(Mouse_tgt, warped_src, i):
    warped_denormalized = inverse_data_normalisation(warped_src, Mouse_tgt.center_list[i])
    raw_x, raw_y = np.asarray(warped_denormalized[:, 0], dtype=int), np.asarray(warped_denormalized[:, 1], dtype=int)
    labeled_nodes = get_list_ROI(raw_x, raw_y, Mouse_tgt.ROI_images[i].T,
                                 image_size=Mouse_tgt.image_sizes[i])
    return labeled_nodes
