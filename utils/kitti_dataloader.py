"""Code for loading KITTI odometry dataset"""

import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import open3d as o3d

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.misc_utils import *


#####################################################################################
# Load poses
#####################################################################################

def transfrom_cam2velo(Tcam):
    R = np.array([7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                  -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
                  ]).reshape(3, 3)
    t = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
    cam2velo = np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

    return Tcam @ cam2velo


def load_poses_from_txt(file_name):
    """
    Modified function from: https://github.com/Huangying-Zhan/kitti-odom-eval/blob/master/kitti_odometry.py
    """
    f = open(file_name, 'r')
    s = f.readlines()
    f.close()
    transforms = {}
    positions = []
    for cnt, line in enumerate(s):
        P = np.eye(4)
        line_split = [float(i) for i in line.split(" ") if i != ""]
        withIdx = len(line_split) == 13
        for row in range(3):
            for col in range(4):
                P[row, col] = line_split[row * 4 + col + withIdx]
        if withIdx:
            frame_idx = line_split[0]
        else:
            frame_idx = cnt
        transforms[frame_idx] = transfrom_cam2velo(P)
        positions.append([P[0, 3], P[2, 3], P[1, 3]])
    return transforms, np.asarray(positions)


def get_delta_pose(transforms):
    rel_transforms = []
    for i in range(len(transforms) - 1):
        w_T_p1 = transforms[i]
        w_T_p2 = transforms[i + 1]

        p1_T_w = T_inv(w_T_p1)
        p1_T_p2 = np.matmul(p1_T_w, w_T_p2)
        rel_transforms.append(p1_T_p2)
    return rel_transforms


#####################################################################################
# Load scans
#####################################################################################


""" Helper functions from https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py """


def load_bin_scan(file):
    """Load and reshape binary file containing single point cloud"""
    scan = np.fromfile(file, dtype=np.float32)
    return scan.reshape((-1, 4))


def yield_bin_scans(bin_files):
    """Generator to load multiple point clouds sequentially"""
    for file in bin_files:
        yield load_bin_scan(file)


def visualize_scan_open3d(ptcloud_xyz, tittle='clouds', colors=[]):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ptcloud_xyz)
    if colors != []:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], tittle)


def visualiza_scan_clouds(raw, filtered, seg, colors=[]):
    pcd_raw = o3d.geometry.PointCloud()
    pcd_raw.points = o3d.utility.Vector3dVector(raw)
    pcd_raw.paint_uniform_color([1, 0, 0])
    pcd_filtered = o3d.geometry.PointCloud()
    pcd_filtered.points = o3d.utility.Vector3dVector(filtered)
    pcd_filtered.paint_uniform_color([0, 1, 0])
    pcd_seg = o3d.geometry.PointCloud()
    pcd_seg.points = o3d.utility.Vector3dVector(seg)
    pcd_seg.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd_raw, pcd_filtered, pcd_seg])


def visualize_sequence_open3d(bin_files, n_scans):
    """Visualize scans using Open3D"""

    scans = yield_bin_scans(bin_files)

    for i in range(n_scans):
        scan = next(scans)
        ptcloud_xyz = scan[:, :-1]
        print(ptcloud_xyz.shape)
        visualize_scan_open3d(ptcloud_xyz)


#####################################################################################
# make points projection
#####################################################################################

def range_projection_points(current_vertex, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50):
    # OverlapNet initial parameters
    # fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50
    """ Project a pointcloud into a spherical projection, range image.
      Args:
        current_vertex: raw point clouds 矩阵构成： 第一列 x 第二列 y 第三列z
      Returns:
        proj_range: projected range image with depth, each pixel contains the corresponding depth   带深度的投影范围图像，每个像素包含深度
        proj_vertex: each pixel contains the corresponding point (x, y, z, 1)                       每个像素包含对应点xyz
        proj_intensity: each pixel contains the corresponding intensity                             每个像素包含对应点的反射强度
        proj_idx: each pixel contains the corresponding index of the point in the raw point cloud   每个像素包含对应点的原始点云索引
  """
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi  # field of view up in radians 弧度制上视角
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians 弧度制下视角
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians 整个视角

    # get depth of all points
    depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)  # 计算每个点云的距离,计算矩阵前三列的2范数
    current_vertex = current_vertex[(depth > 0) & (depth < max_range)]  # get rid of [0, 0, 0] points
    depth = depth[(depth > 0) & (depth < max_range)]

    # get scan components
    scan_x = current_vertex[:, 0]
    scan_y = current_vertex[:, 1]
    scan_z = current_vertex[:, 2]

    # get angles of all points
    # 计算偏航角和俯仰角
    # 偏航角为偏离x轴方向，运动正向为x轴正向，y轴负向，arctan(y/x)
    yaw = -np.arctan2(scan_y, scan_x)
    # 俯仰角为z轴和深度的反正弦，arcsin(z/d)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    # 按照偏航角和俯仰角进行球面投影
    # 偏航角在-180～180，进行非负化和归一化处理
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    # 加上最低视角进行非负化，再除以总视角进行归一化，再被1减，符合图像坐标系
    # 如果激光雷达的扫描线数不是均匀的，则投影也不是均匀的，需要注意
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    # 恢复尺度
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)  # 对整个投影向下取整
    proj_x = np.minimum(proj_W - 1, proj_x)  # 确保上下界
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    # order in decreasing depth
    order = np.argsort(depth)[::-1]  # 获得按距离的降序排列的序号
    depth = depth[order]


    proj_y = proj_y[order]  # 按照距离远近降序排列
    proj_x = proj_x[order]

    scan_x = scan_x[order]
    scan_y = scan_y[order]
    scan_z = scan_z[order]  # 将xyz的顺序与深度顺序对齐

    indices = np.arange(depth.shape[0])
    indices = indices[order]
    # 初始化伪图像
    proj_range = np.full((proj_H, proj_W), -1,
                         dtype=np.float32)  # [H,W] range (-1 is no data)
    proj_vertex = np.full((proj_H, proj_W, 4), -1,
                          dtype=np.float32)  # [H,W] index (-1 is no data)
    proj_idx = np.full((proj_H, proj_W), -1,
                       dtype=np.int32)  # [H,W] index (-1 is no data)

    # 按照距离顺序进行图像点赋值，因为分辨率的压缩，会存在取整后重复的点，按照距离降序赋值的话，同一点选择距离最近的值作为代表
    proj_range[proj_y, proj_x] = depth
    proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z, np.ones(len(scan_x))]).T
    proj_idx[proj_y, proj_x] = indices

    return proj_range, proj_vertex, proj_idx


def range_projection_colorseg(current_vertex, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50):
    # OverlapNet initial parameters
    # fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50
    """ Project a pointcloud into a spherical projection, range image.
      Args:
        current_vertex: raw point clouds 矩阵构成： 第一列 x 第二列 y 第三列z 第四列反射强度
      Returns:
        proj_range: projected range image with depth, each pixel contains the corresponding depth   带深度的投影范围图像，每个像素包含深度
        proj_vertex: each pixel contains the corresponding point (x, y, z, 1)                       每个像素包含对应点xyz
        proj_intensity: each pixel contains the corresponding intensity                             每个像素包含对应点的反射强度
        proj_idx: each pixel contains the corresponding index of the point in the raw point cloud   每个像素包含对应点的原始点云索引
  """
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi  # field of view up in radians 弧度制上视角
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians 弧度制下视角
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians 整个视角

    # get depth of all points
    depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)  # 计算每个点云的距离,计算矩阵前三列的2范数
    current_vertex = current_vertex[(depth > 0) & (depth < max_range)]  # get rid of [0, 0, 0] points
    depth = depth[(depth > 0) & (depth < max_range)]

    # get scan components
    scan_x = current_vertex[:, 0]
    scan_y = current_vertex[:, 1]
    scan_z = current_vertex[:, 2]
    color = current_vertex[:, 3]
    centroid = current_vertex[:, 4]
    # print("color", color)
    # get angles of all points
    # 计算偏航角和俯仰角
    # 偏航角为偏离x轴方向，运动正向为x轴正向，y轴负向，arctan(y/x)
    yaw = -np.arctan2(scan_y, scan_x)
    # 俯仰角为z轴和深度的反正弦，arcsin(z/d)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    # 按照偏航角和俯仰角进行球面投影
    # 偏航角在-180～180，进行非负化和归一化处理
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    # 加上最低视角进行非负化，再除以总视角进行归一化，再被1减，符合图像坐标系
    # 如果激光雷达的扫描线数不是均匀的，则投影也不是均匀的，需要注意
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    # 恢复尺度
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)  # 对整个投影向下取整
    proj_x = np.minimum(proj_W - 1, proj_x)  # 确保上下界
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    # order in decreasing depth
    order = np.argsort(depth)[::-1]  # 获得按距离的降序排列的序号
    depth = depth[order]

    color = color[order]  # 反射强度和距离相关，一样的序号
    centroid = centroid[order]
    # print("here color type:", type(color))
    proj_y = proj_y[order]  # 按照距离远近降序排列
    proj_x = proj_x[order]

    scan_x = scan_x[order]
    scan_y = scan_y[order]
    scan_z = scan_z[order]  # 将xyz的顺序与深度顺序对齐

    indices = np.arange(depth.shape[0])
    indices = indices[order]
    # 初始化伪图像
    proj_range = np.full((proj_H, proj_W), -1,
                         dtype=np.float32)  # [H,W] range (-1 is no data)
    proj_vertex = np.full((proj_H, proj_W, 4), -1,
                          dtype=np.float32)  # [H,W] index (-1 is no data)
    proj_idx = np.full((proj_H, proj_W), -1,
                       dtype=np.int32)  # [H,W] index (-1 is no data)
    proj_color = np.full((proj_H, proj_W), 0,
                             dtype=np.float32)  # [H,W] index (-1 is no data)
    proj_centroid = np.full((proj_H, proj_W), 0,
                         dtype=np.float32)  # [H,W] index (-1 is no data)
    # 按照距离顺序进行图像点赋值，因为分辨率的压缩，会存在取整后重复的点，按照距离降序赋值的话，同一点选择距离最近的值作为代表
    # print("depth", depth)
    proj_range[proj_y, proj_x] = depth
    proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z, np.ones(len(scan_x))]).T
    proj_idx[proj_y, proj_x] = indices
    proj_color[proj_y, proj_x] = color
    proj_centroid[proj_y, proj_x] = centroid
    return proj_range, proj_vertex, proj_color, proj_idx, proj_centroid
#####################################################################################
# Load timestamps
#####################################################################################


def load_timestamps(file_name):
    # file_name = data_dir + '/times.txt'
    file1 = open(file_name, 'r+')
    stimes_list = file1.readlines()
    s_exp_list = np.asarray([float(t[-4:-1]) for t in stimes_list])
    times_list = np.asarray([float(t[:-2]) for t in stimes_list])
    times_listn = [times_list[t] * (10 ** (s_exp_list[t]))
                   for t in range(len(times_list))]
    file1.close()
    return times_listn


#####################################################################################
# Test
#####################################################################################


if __name__ == "__main__":

    # Set the dataset location here:
    basedir = '/mnt/088A6CBB8A6CA742/Datasets/Kitti/dataset/'

    ##################
    # Test poses

    fig, axs = plt.subplots(4, 6, constrained_layout=True)
    fig.suptitle('KITTI sequences', fontsize=16)
    for i in range(22):
        sequence = str(i)
        if i < 10:
            sequence = '0' + str(i)
        sequence_path = basedir + 'sequences/' + sequence + '/'
        poses_file = sorted(
            glob.glob(os.path.join(sequence_path, 'poses.txt')))
        _, positions = load_poses_from_txt(poses_file[0])
        print('seq: ', sequence, 'len', len(positions))

        axs[i // 6, i % 6].plot(positions[:, 0], positions[:, 1])
        axs[i // 6, i % 6].set_title('seq: ' + sequence + 'len' + str(len(positions)))

    plt.show()

    ##################
    # Test scans

    sequence = '00'
    sequence_path = basedir + 'sequences/' + sequence + '/'
    bin_files = sorted(glob.glob(os.path.join(
        sequence_path, 'velodyne', '*.bin')))
    # Visualize some scans
    visualize_sequence_open3d(bin_files, 2)

    ##################
    # Test timestamps
    timestamps_file = basedir + 'sequences/' + sequence + '/times.txt'
    timestamps = load_timestamps(timestamps_file)
    print("Start time (s): ", timestamps[0])
    print("End time (s): ", timestamps[-1])

    print('Test complete.')
