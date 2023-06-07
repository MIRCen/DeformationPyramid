import numpy as np
import open3d as o3d
import copy
import scipy
from registration_functions import apply_shape_transfer
def print_points_cloud_information(pc):
    print(pc)
    print(f"points : {np.asarray(pc.points)}")
    print(f"VoxelGrid : {np.asarray(pc.VoxelGrid)}")
    print(f"Colors : {np.asarray(pc.colors)}")

def visualize_label(features, labeled_nodes, region=None):

    labeled_nodes = labeled_nodes%100 #[label % 100 for label in labeled_nodes]
    if region is not None:
        filtered_array =labeled_nodes==2
        print(filtered_array)
        labeled_nodes=labeled_nodes[filtered_array]
        features=features[filtered_array]

    features_pc = o3d.geometry.PointCloud()
    features_pc.points = o3d.utility.Vector3dVector(features)
    np_col = np.zeros((len(labeled_nodes), 3), dtype=float)
    np_col[:, 0] = [((l * 2) % 40) / 40 for l in labeled_nodes]
    np_col[:, 1] = [((l * 20) % 40) / 40 for l in labeled_nodes]
    np_col[:, 2] = [((l * 37) % 40) / 40 for l in labeled_nodes]
    col = o3d.utility.Vector3dVector(np_col)
    features_pc.colors = col
    features_pc = features_pc.voxel_down_sample(voxel_size=0.01)
    o3d.visualization.draw_geometries([features_pc])


def visualize_registration(tgt_pcd, src_pcd, warped_pcd):
    warped_pc = o3d.geometry.PointCloud()
    warped_pc.points = o3d.utility.Vector3dVector(warped_pcd)

    tgt_pc = o3d.geometry.PointCloud()
    tgt_pc.points = o3d.utility.Vector3dVector(tgt_pcd)

    src_pc = o3d.geometry.PointCloud()
    src_pc.points = o3d.utility.Vector3dVector(src_pcd)

    tgt_pc.paint_uniform_color([0.4, 0.4, 1])
    warped_copy=copy.deepcopy(warped_pc)
    warped_copy.paint_uniform_color([0.4,1, 0.4])
    src_pc.paint_uniform_color([1, 0.4, 0.4])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Target (blue), Warped ()', width=960, height=660, left=0, top=0)
    vis.add_geometry(tgt_pc)
    vis.add_geometry(warped_copy)

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name='Target (), Source ()', width=960, height=660, left=1100, top=0)
    vis2.add_geometry(tgt_pc)
    vis2.add_geometry(src_pc)

    vis3 = o3d.visualization.Visualizer()
    vis3.create_window(window_name='Warped(), Source ()', width=960, height=660, left=0, top=750)
    vis3.add_geometry(src_pc)
    vis3.add_geometry(warped_copy)


    vis4 = o3d.visualization.Visualizer()
    vis4.create_window(window_name='Transformation', width=960, height=660, left=1100, top=750)
    vis4.get_view_control().set_zoom(0.1)

    vis4.add_geometry(warped_pc)

    while True:
        if not vis.poll_events() or not vis2.poll_events() or not vis3.poll_events() or not vis4.poll_events():
            break
        vis.update_renderer()
        vis2.update_renderer()
        vis3.update_renderer()
        vis4.update_renderer()


def visualize_deformation(NDP,min_level=0):
    X,Y =np.mgrid[-1.2:1.2:0.01, -1.2:1.2:0.01]
    xy = np.vstack((X.flatten(), Y.flatten())).T

    grid = np.concatenate((xy, np.zeros((len(xy),1))),axis=1)
    transformed_grid=apply_shape_transfer(NDP, grid.astype('float32'), min_level=min_level)
    distance_grid=np.sum(np.sqrt((transformed_grid-grid)**2),axis=1)

    grid_pc = o3d.geometry.PointCloud()
    grid_pc.points = o3d.utility.Vector3dVector(grid)
    grid_pc.paint_uniform_color([0.9, 0.9, 0.9])

    transformed_grid[:,2]=-distance_grid
    transformed_grid_pc = o3d.geometry.PointCloud()
    np_col = np.ones((len(distance_grid), 3), dtype=float)
    np_col[:, 0] = (distance_grid-distance_grid.min())/(distance_grid.max()-distance_grid.min())
    d1=np.clip(distance_grid,0,0.5)*2
    np_col[:, 1] = 1-(d1-d1.min())/(d1.max()-d1.min())
    d2=np.clip(distance_grid,0,0.05)*20
    np_col[:, 2] = 1-(d2-d2.min())/(d2.max()-d2.min())
    col = o3d.utility.Vector3dVector(np_col)
    transformed_grid_pc.colors = col
    transformed_grid_pc.points = o3d.utility.Vector3dVector(transformed_grid)

    o3d.visualization.draw_geometries([transformed_grid_pc])#, grid_pc])

    return transformed_grid,xy





"""
    while True:
        vis.update_geometry()
        if not vis.poll_events():
            break
        vis.update_renderer()

        vis_result.update_geometry()
        if not vis_result.poll_events():
            break
        vis_result.update_renderer()
"""

