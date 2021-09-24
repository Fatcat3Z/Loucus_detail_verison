"""Code for extracting Euclidean segments from a point cloud."""

import numpy as np
from numpy.core.numeric import count_nonzero
import pcl
from pcl import pcl_visualization
import os
import sys
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.kitti_dataloader import visualize_scan_open3d
from utils.kitti_dataloader import visualiza_scan_clouds
from utils.kitti_dataloader import range_projection_points
from utils.kitti_dataloader import range_projection_colorseg

# 提取欧几里德点云簇
def extract_cluster_indices(cloud_filtered, seg_params):
    tree = cloud_filtered.make_kdtree()
    ec = cloud_filtered.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(seg_params['c_tolerence'])
    ec.set_MinClusterSize(seg_params['c_min_size'])
    ec.set_MaxClusterSize(seg_params['c_max_size'])
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()
    return cluster_indices


def extract_segments(scan, seg_params):
    if seg_params['visualize']:
        visualize_scan_open3d(scan, 'raw clouds')

    # Ground Plane Removal
    # PointCloud转化为pcl 格式类型PCD
    # 点云进行平面分割
    cloud = pcl.PointCloud(scan)
    points_raw = np.zeros((cloud.size, 3), dtype=np.float32)

    for i, indices in enumerate(cloud):
        # print(indices)
        points_raw[i][0] = indices[0]
        points_raw[i][1] = indices[1]
        points_raw[i][2] = indices[2]

    seg = cloud.make_segmenter_normals(ksearch=50)
    seg.set_optimize_coefficients(True)
    # SACMODEL_NORMAL_PLANE平面标志 SACMODEL_CYLINDER柱面标志
    seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(seg_params['g_dist_thresh'])
    seg.set_normal_distance_weight(seg_params['g_normal_dist_weight'])
    seg.set_max_iterations(100)
    indices, coefficients = seg.segment()
    # print(indices)
    # crop_xyz 除去地面的点云
    crop_xyz = np.asarray(cloud)
    before_clouds = crop_xyz.size
    # print(before_clouds)
    for k, indice in enumerate(indices):
        # k 滤除地面后点云重新开始的序号
        # indice 保留的点云原始序号
        # 将提取的全部地面点云的z值设置为-20
        crop_xyz[indice][2] = -20.0
    # 根据传感器的高度进一步筛选,传感器为坐标轴中心，所以地面上物体为 z > sensor_height
    crop_xyz = crop_xyz[crop_xyz[:, -1] > -seg_params['g_height']]
    # Voxel filter (optional)  点云体素化
    ds_f = seg_params['ds_factor']
    if ds_f > 0.01:
        cloud = pcl.PointCloud(crop_xyz)
        vg = cloud.make_voxel_grid_filter()
        vg.set_leaf_size(ds_f, ds_f, ds_f)
        cloud_filtered = vg.filter()
    else:
        cloud_filtered = pcl.PointCloud(crop_xyz)

    if seg_params['visualize']:
        visualize_scan_open3d(cloud_filtered, 'filtered cloud')

    # Euclidean Cluster Extraction
    cluster_indices = extract_cluster_indices(cloud_filtered, seg_params)

    if seg_params['visualize']:
        print('cluster_indices : ', np.shape(cluster_indices))

    segments = []
    points_database = []
    colours_database = []
    point_with_color_database = []
    init = False
    # 
    for j, indices in enumerate(cluster_indices):
        points = np.zeros((len(indices), 3), dtype=np.float32)
        # 存储每个段中的点
        for k, indice in enumerate(indices):
            points[k][0] = cloud_filtered[indice][0]
            points[k][1] = cloud_filtered[indice][1]
            points[k][2] = cloud_filtered[indice][2]

        # Additional filtering step to remove flat(ground-plane) segments 去除平坦的段平面
        x_diff = (max(points[:, 0]) - min(points[:, 0]))
        y_diff = (max(points[:, 1]) - min(points[:, 1]))
        z_diff = (max(points[:, 2]) - min(points[:, 2]))
        # 识别分割出的段中的便平面，识别方法最大的x或y差值/z差值，小于预设阈值，即俯仰角或滚转角小于一定角度
        # 分割出的段就是扁平的面
        if (not seg_params['filter_flat_seg']) or (max(x_diff, y_diff) / z_diff < seg_params['horizontal_ratio']):
            segments.append(points)
            colour = np.random.random_sample((1))
            if init:
                # vstack 按行串联数组
                points_database = np.vstack((points_database, points))
                colour = np.tile(colour, (len(indices), 1))
                colours_database = np.vstack((colours_database, colour))

            else:
                points_database = points
                colour = np.tile(colour, (len(indices), 1))
                colours_database = colour
                init = True
    point_with_color_database = np.hstack((points_database, colours_database))
    if seg_params['visualize']:
        visualize_scan_open3d(points_database, 'seqment clouds', colours_database)

    proj_range_seg, _, color_segmets, _ = range_projection_colorseg(point_with_color_database)
    proj_range_raw, _, _ = range_projection_points(points_raw)

    fig, axs = plt.subplots(3, figsize=(6, 4))
    axs[0].set_title('range_data_raw')
    axs[0].imshow(proj_range_raw)
    axs[0].set_axis_off()

    axs[1].set_title('range_data_segments')
    axs[1].imshow(proj_range_seg)
    axs[1].set_axis_off()

    axs[2].set_title('range_data_segments_colored')
    axs[2].imshow(color_segmets)
    axs[2].set_axis_off()

    plt.suptitle('Difference between raw and segments data')
    plt.show()

    # visualiza_scan_clouds(cloud, cloud_filtered, points_database, colours_database)
    return segments


def extract_segments_2d(scan, seg_params):
    if seg_params['visualize']:
        visualize_scan_open3d(scan, 'raw clouds')

    # Ground Plane Removal
    # PointCloud转化为pcl 格式类型PCD
    # 点云进行平面分割
    cloud = pcl.PointCloud(scan)
    points_raw = np.zeros((cloud.size, 3), dtype=np.float32)

    for i, indices in enumerate(cloud):
        # print(indices)
        points_raw[i][0] = indices[0]
        points_raw[i][1] = indices[1]
        points_raw[i][2] = indices[2]

    seg = cloud.make_segmenter_normals(ksearch=50)
    seg.set_optimize_coefficients(True)
    # SACMODEL_NORMAL_PLANE平面标志 SACMODEL_CYLINDER柱面标志
    seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(seg_params['g_dist_thresh'])
    seg.set_normal_distance_weight(seg_params['g_normal_dist_weight'])
    seg.set_max_iterations(100)
    indices, coefficients = seg.segment()
    # print(indices)
    # crop_xyz 除去地面的点云
    crop_xyz = np.asarray(cloud)
    before_clouds = crop_xyz.size
    # print(before_clouds)
    for k, indice in enumerate(indices):
        # k 滤除地面后点云重新开始的序号
        # indice 保留的点云原始序号
        # 将提取的全部地面点云的z值设置为-20
        crop_xyz[indice][2] = -20.0
    # 根据传感器的高度进一步筛选,传感器为坐标轴中心，所以地面上物体为 z > sensor_height
    crop_xyz = crop_xyz[crop_xyz[:, -1] > -seg_params['g_height']]
    # Voxel filter (optional)  点云体素化
    ds_f = seg_params['ds_factor']
    if ds_f > 0.01:
        cloud = pcl.PointCloud(crop_xyz)
        vg = cloud.make_voxel_grid_filter()
        vg.set_leaf_size(ds_f, ds_f, ds_f)
        cloud_filtered = vg.filter()
    else:
        cloud_filtered = pcl.PointCloud(crop_xyz)

    if seg_params['visualize']:
        visualize_scan_open3d(cloud_filtered, 'filtered cloud')

    # Euclidean Cluster Extraction
    cluster_indices = extract_cluster_indices(cloud_filtered, seg_params)

    if seg_params['visualize']:
        print('cluster_indices : ', np.shape(cluster_indices))

    points_database = []
    colours_database = []
    centroid_database = []
    init = False
    #
    for j, indices in enumerate(cluster_indices):
        points = np.zeros((len(indices), 3), dtype=np.float32)
        # 存储每个段中的点
        for k, indice in enumerate(indices):
            points[k][0] = cloud_filtered[indice][0]
            points[k][1] = cloud_filtered[indice][1]
            points[k][2] = cloud_filtered[indice][2]

        # Additional filtering step to remove flat(ground-plane) segments 去除平坦的段平面
        x_diff = (max(points[:, 0]) - min(points[:, 0]))
        y_diff = (max(points[:, 1]) - min(points[:, 1]))
        z_diff = (max(points[:, 2]) - min(points[:, 2]))
        # 识别分割出的段中的便平面，识别方法最大的x或y差值/z差值，小于预设阈值，即俯仰角或滚转角小于一定角度
        # 分割出的段就是扁平的面
        #
        if (not seg_params['filter_flat_seg']) or (max(x_diff, y_diff) / z_diff < seg_params['horizontal_ratio']):
            # ji suan mei ge duan de zhi xin
            centroid = np.mean(points, axis=0)
            depth = np.linalg.norm(centroid, 2)
            depth = depth[(depth > 0) & (depth < 50)]
            depth = np.array([depth])
            colour = np.random.random_sample((1))
            if init:
                # vstack 按行串联数组
                points_database = np.vstack((points_database, points))
                colour = np.tile(colour, (len(indices), 1))
                colours_database = np.vstack((colours_database, colour))
                depth = np.tile(depth, (len(indices), 1))
                centroid_database = np.vstack((centroid_database, depth))

            else:
                points_database = points
                colour = np.tile(colour, (len(indices), 1))
                colours_database = colour
                depth = np.tile(depth, (len(indices), 1))
                centroid_database = depth
                init = True

    point_with_color_database = np.hstack((points_database, colours_database,centroid_database))

    print(point_with_color_database.size)
    # if seg_params['visualize']:
    #     visualize_scan_open3d(points_database, 'seqment clouds', colours_database)
    #
    proj_range_seg, _, color_segmets, _, proj_centroid_range = range_projection_colorseg(point_with_color_database)
    proj_range_raw, _, _ = range_projection_points(points_raw)
    #
    fig, axs = plt.subplots(4, figsize=(6, 4))
    axs[0].set_title('range_data_raw')
    axs[0].imshow(proj_range_raw)
    axs[0].set_axis_off()

    axs[1].set_title('range_data_segments')
    axs[1].imshow(proj_range_seg)
    axs[1].set_axis_off()

    axs[2].set_title('range_data_segments_colored')
    axs[2].imshow(color_segmets)
    axs[2].set_axis_off()

    axs[3].set_title('centroid_range_data_segments')
    axs[3].imshow(proj_centroid_range)
    axs[3].set_axis_off()

    plt.suptitle('Difference between raw and segments data')
    plt.show()

    # visualiza_scan_clouds(cloud, cloud_filtered, points_database, colours_database)
    return proj_range_raw

def get_segments(scan, seg_params):
    segments = extract_segments(scan, seg_params)

    # Handling rare degenerate scenes 处理稀有的退化场景
    c_tolerence = seg_params['c_tolerence']
    while seg_params['enforce_min_seg_count'] and len(segments) < seg_params['min_seg_count']:
        seg_params['c_tolerence'] -= 0.01
        if seg_params['c_tolerence'] < 0.01:
            break
        segments = extract_segments(scan, seg_params)
    seg_params['c_tolerence'] = c_tolerence
    return segments


#####################################################################################
# Test
#####################################################################################


if __name__ == "__main__":

    import glob
    import yaml
    from utils.kitti_dataloader import yield_bin_scans
    from utils.misc_utils import save_pickle

    seq = '07'

    cfg_file = open('config.yml', 'r')
    cfg_params = yaml.load(cfg_file, Loader=yaml.FullLoader)
    seg_params = cfg_params['segmentation']

    basedir = cfg_params['paths']['KITTI_dataset']
    sequence_path = basedir + 'sequences/' + seq + '/'
    bin_files = sorted(glob.glob(os.path.join(
        sequence_path, 'velodyne', '*.bin')))

    scans = yield_bin_scans(bin_files)
    segments_database = []

    # scan = next(scans)
    # cloud = pcl.PointCloud(scan[:, :-1])
    # # only get x y z, remove identity
    # segments = extract_segments(scan[:, :-1], seg_params)

    for i in range(10):
        scan = next(scans)
        segments = extract_segments_2d(scan[:, :-1], seg_params)
        print('Extracted segments: ', np.shape(segments))
        segments_database.append(segments)

    # save_dir = cfg_params['paths']['save_dir'] + seq
    # save_pickle(segments_database, save_dir +
    #             '/segments_database.pickle')
