import pandas as pd
import PIL.Image as Image
import numpy as np
import os


def get_list_ROI(list_centroids_X, list_centroids_Y, ROI_img, image_size=None):
    """
    project point coordinatinates to ROI_img to get point labels

    :param list_centroids_X:
    :param list_centroids_Y:
    :param ROI_img:
    :param image_size:
    :return: list of labels
    """
    ROI_image_size = ROI_img.shape
    list_cell_ROI = []
    for centroid_X, centroid_Y in zip(list_centroids_X, list_centroids_Y):

        index_x= max(min(int(centroid_X * (ROI_image_size[0] / image_size[0])),ROI_image_size[0]-1),0)
        index_y=max(min(int(centroid_Y * (ROI_image_size[1] / image_size[1])),ROI_image_size[1]-1),0)
        list_cell_ROI.append(ROI_img[index_x, index_y])
    return np.asarray(list_cell_ROI)

def data_transformation(features):
    features[:, 1] = features[:, 1].max() - features[:, 1]
    src_percentile_99 = np.percentile(features[:, 0:2], 99, axis=0)
    src_percentile_01 = np.percentile(features[:, 0:2], 1, axis=0)
    src_middle = (src_percentile_99 + src_percentile_01) / 2
    features[:, 0:2] = (features[:, 0:2] - src_middle) / [20000, 15000]
    return features, src_middle



def inverse_data_normalisation(features_normalized, middle):
     features=features_normalized.copy()
     features[:, 0:2] = features[:, 0:2]*[20000, 15000] + middle
     features[:, 1] = -features[:, 1]
     return features



def read_aims_features(image_urls, selected_features=['mc_x','mc_y']):
    """
    :param image_urls:
    :param selected_features:
    :return:
    """
    list_features=[]
    for url in image_urls:
        data = pd.read_csv(url, sep =';')
        data=data[['mc_x','mc_y']]
        data=data.assign(mc_z=0)
        numpy_data = data.to_numpy(dtype=np.float32)
        list_features.append(numpy_data)
    return list_features

def read_image(img_url):
    return np.asarray(Image.open(img_url))

def read_file_info(data_training_info_slices):
    from pandas_ods_reader import read_ods

    test_a=os.path.isfile(data_training_info_slices)
    df = read_ods(data_training_info_slices, 1)
    rotations = df["rotation"].to_numpy()
    slices_numeros = df["numero"].to_numpy()
    dim_x_image = df["dim_x_image"].to_numpy()
    dim_y_image = df["dim_y_image"].to_numpy()
    return slices_numeros, rotations,[(x,y) for x,y in zip(dim_x_image,dim_y_image)]
