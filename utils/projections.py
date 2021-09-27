import numpy as np
from open3d import *
import matplotlib.pyplot as plt
from scipy.spatial import distance


#####################################################################################
# make points projection
#####################################################################################
def generate_color_ranged(color, segments):
    color_ranged = []
    classes = []
    # 根据段特征数进行颜色分级
    for i in range(segments):
        class_color = (i + 1) / segments
        classes.append(class_color)
    all_color = [color[0]]
    # 按照已经降序排列的距离进行挑选
    for i in range(len(color)):
        if color[i] not in all_color:
            all_color.append([color[i]])
    for i in range(len(color)):
        for j in range(len(all_color)):
            if color[i] == all_color[j]:
                color_ranged.append(classes[j])
    return color_ranged


def spatial_triangle_average(centriods, topk):
    spatial_segments = []
    centriod_points_x = [centriods[0][0]]
    centriod_points_y = [centriods[0][1]]
    centriod_points_z = [centriods[0][2]]
    for i in range(len(centriods)):
        if centriods[i][0] not in centriod_points_x:
            centriod_points_x.append(centriods[i][0])
            centriod_points_y.append(centriods[i][1])
            centriod_points_z.append(centriods[i][2])
    segments_classes = len(centriod_points_x)
    # print("segments_classes", segments_classes)
    dist_mat = np.zeros((segments_classes, segments_classes))
    # 计算各个质心之间的距离
    for j in range(segments_classes):
        p_j_x = centriod_points_x[j]
        p_j_y = centriod_points_y[j]
        p_j_z = centriod_points_z[j]
        for k in range(segments_classes):
            if j >= k:  # Only need to calculate upper triangle.
                continue
            p_k_x = centriod_points_x[k]
            p_k_y = centriod_points_y[k]
            p_k_z = centriod_points_z[k]
            dist = np.sqrt(np.square(p_j_x - p_k_x) + np.square(p_j_y - p_k_y) + np.square(p_j_z - p_k_z))
            dist_mat[j][k] = dist
            dist_mat[k][j] = dist
    min_dists = []
    min_dist_ids = []
    # 选出最近的k个质心
    for s in range(segments_classes):
        dist_vec = dist_mat[s]
        min_dist_id = dist_vec.argsort()[1:topk + 1]
        min_dist_ids.append(min_dist_id)
        min_dists.append(dist_vec[min_dist_id])
    area = np.zeros((segments_classes, topk - 2))
    area_factor = np.zeros((segments_classes, topk - 2))
    area_average = []
    # 计算各质心连线的面积
    for i in range(segments_classes):
        points = np.zeros((3, 3))
        points[0] = [centriod_points_x[i], centriod_points_y[i], centriod_points_z[i]]
        points[1] = [centriod_points_x[min_dist_ids[i][1]],
                     centriod_points_y[min_dist_ids[i][1]],
                     centriod_points_z[min_dist_ids[i][1]]]
        for j in range(2, topk):
            points[2] = [centriod_points_x[min_dist_ids[i][j]],
                         centriod_points_y[min_dist_ids[i][j]],
                         centriod_points_z[min_dist_ids[i][j]]]
            # factor = min_dists[i][2] / min_dists[i][j]
            # area[i][j - 2] = factor * calculate_area_of_triangle(points)
            area[i][j - 2] = calculate_area_of_triangle(points)
            factor = -0.1
            area_factor[i][j - 2] = np.exp(factor * area[i][j - 2])

        # 算加权面积
        area_sum = np.sum(area_factor[i])
        spatial_area = 0
        for j in range(2, topk):
            spatial_area += (area_factor[i][j-2] / area_sum) * area[i][j - 2]
        area_average.append(spatial_area)
        # area_average.append(np.mean(area[i]))
    # print(area_average)
    # 把对应质心的值改成加权面积,并归一化
    max_area = np.max(area_average)
    for i in range(len(centriods)):
        for j in range(segments_classes):
            if centriods[i][0] == centriod_points_x[j]:
                spatial_segments.append(area_average[j] / max_area)
    return spatial_segments


def calculate_area_of_triangle(points):
    length = []
    for i in range(len(points)):
        tmplen = np.sqrt(points[i][0] * points[i][0] + points[i][1] * points[i][1] + points[i][2] * points[i][2])
        length.append(tmplen)
    p = 0.5 * (length[0] + length[1] + length[2])
    halensum = p * (p - length[0]) * (p - length[1]) * (p - length[2])
    # 如果出现三点一线的情况
    if halensum < 0:
        res = 0
    else:
        res = np.sqrt(halensum)
    # 海伦公式算面积
    return res


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


def range_projection_colorseg(current_vertex, segments_num, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900,
                              max_range=50):
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
    # centroid = current_vertex[:, 4]
    centroid_points = current_vertex[:, 4:7]
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

    color = color[order] / np.max(color)  # 反射强度和距离相关，一样的序号
    color_ranged = generate_color_ranged(color, segments_num)
    # centroid = centroid[order]
    centroid_points = centroid_points[order]
    spatial_area = spatial_triangle_average(centroid_points, 5)
    final_spatial = [color, color_ranged, spatial_area]
    final_spatial = np.transpose(final_spatial)
    # print("color size", color.size)
    # print("color_ranged size", len(color_ranged))
    # print("spatial_area size", len(spatial_area))

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
    proj_color = np.full((proj_H, proj_W, 3), 0,
                         dtype=np.float32)  # [H,W] index (-1 is no data)
    proj_centroid = np.full((proj_H, proj_W), 0,
                            dtype=np.float32)  # [H,W] index (-1 is no data)
    # 按照距离顺序进行图像点赋值，因为分辨率的压缩，会存在取整后重复的点，按照距离降序赋值的话，同一点选择距离最近的值作为代表
    # print("depth", depth)
    proj_range[proj_y, proj_x] = depth
    proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z, np.ones(len(scan_x))]).T
    proj_idx[proj_y, proj_x] = indices
    proj_color[proj_y, proj_x] = final_spatial
    proj_centroid[proj_y, proj_x] = spatial_area
    return proj_range, proj_vertex, proj_color, proj_idx, proj_centroid


# 生成点云法向量图
def gen_normal_map(current_range, current_vertex, proj_H=64, proj_W=900):
    """ Generate a normal image given the range projection of a point cloud.
        Args:
          current_range:  range projection of a point cloud, each pixel contains the corresponding depth
          current_vertex: range projection of a point cloud,
                          each pixel contains the corresponding point (x, y, z, 1)
        Returns:
          normal_data: each pixel contains the corresponding normal
    """
    normal_data = np.full((proj_H, proj_W, 3), -1, dtype=np.float32)

    # iterate over all pixels in the range image
    for x in range(proj_W):
        for y in range(proj_H - 1):
            p = current_vertex[y, x][:3]
            depth = current_range[y, x]

            if depth > 0:
                wrap_x = wrap(x + 1, proj_W)
                u = current_vertex[y, wrap_x][:3]
                u_depth = current_range[y, wrap_x]
                if u_depth <= 0:
                    continue

                v = current_vertex[y + 1, x][:3]
                v_depth = current_range[y + 1, x]
                if v_depth <= 0:
                    continue
                # 法向量的计算：选取下一个点的连线与右侧点的连线组成的外积
                u_norm = (u - p) / np.linalg.norm(u - p)
                v_norm = (v - p) / np.linalg.norm(v - p)

                w = np.cross(v_norm, u_norm)
                norm = np.linalg.norm(w)
                if norm > 0:
                    normal = w / norm
                    normal_data[y, x] = normal

    return normal_data


def wrap(x, dim):
    """ Wrap the boarder of the range image.
  """
    value = x
    if value >= dim:
        value = (value - dim)
    if value < 0:
        value = (value + dim)
    return value


#####################################################################################
# make scancontext
#####################################################################################
class ScanContext:
    # static variables
    # sector_res = np.array([45, 90, 180, 360, 720])
    # ring_res = np.array([10, 20, 40, 80, 160])
    # sector_res = np.array([60])
    # ring_res = np.array([20])

    def __init__(self, points, seg_params):
        self.SCs = []
        self.points = points
        self.lidar_height = seg_params['g_height']
        self.max_range = seg_params['max_range']
        self.downcell_size = seg_params['downcell_size']
        self.viz = seg_params['visualize']
        self.num_sector = seg_params['num_sector']
        self.num_ring = seg_params['num_ring']
        self.scancontexts = self.genSCs()

    # 右手螺旋方向角度变化
    def xy2theta(self, x, y):
        yaw = -np.arctan2(y, x)
        theta = 180 * (yaw / np.pi + 1.0)
        # if (x >= 0 and y >= 0):
        #     theta = 180 / np.pi * np.arctan(y / x)
        # if (x < 0 and y >= 0):
        #     theta = 180 - ((180 / np.pi) * np.arctan(y / (-x)))
        # if (x < 0 and y < 0):
        #     theta = 180 + ((180 / np.pi) * np.arctan(y / x))
        # if (x >= 0 and y < 0):
        #     theta = 360 - ((180 / np.pi) * np.arctan((-y) / x))
        return theta

    def pt2rs(self, point, gap_ring, gap_sector, num_ring, num_sector):
        x = point[0]
        y = point[1]
        z = point[2]

        if (x == 0.0):
            x = 0.001
        if (y == 0.0):
            y = 0.001

        theta = self.xy2theta(x, y)
        faraway = np.sqrt(x * x + y * y)
        # 整除距离和角度分辨率，获得编码序号
        idx_ring = np.divmod(faraway, gap_ring)[0]
        idx_sector = np.divmod(theta, gap_sector)[0]

        if (idx_ring >= num_ring):
            idx_ring = num_ring - 1  # python starts with 0 and ends with N-1

        return int(idx_ring), int(idx_sector)

    # 把点云转化为scancontext
    def ptcloud2sc(self, ptcloud, num_sector, num_ring, max_length):

        num_points = ptcloud.shape[0]
        # max_length 最远有效距离， num_ring 分隔的间隔数， num_sector 一圈中的角度间隔
        # 距离分辨率
        gap_ring = max_length / num_ring
        # 角度分辨率
        gap_sector = 360 / num_sector

        enough_large = 1000
        # 记录区域内的全部点，最后选取最大的高度
        sc_storage = np.zeros([enough_large, num_ring, num_sector])
        sc_counter = np.zeros([num_ring, num_sector])

        for pt_idx in range(num_points):

            point = ptcloud[pt_idx, :]
            point_height = point[2] + self.lidar_height

            idx_ring, idx_sector = self.pt2rs(point, gap_ring, gap_sector, num_ring, num_sector)

            if sc_counter[idx_ring, idx_sector] >= enough_large:
                continue
            sc_storage[int(sc_counter[idx_ring, idx_sector]), idx_ring, idx_sector] = point_height
            sc_counter[idx_ring, idx_sector] = sc_counter[idx_ring, idx_sector] + 1

        sc = np.amax(sc_storage, axis=0)

        return sc

    def genSCs(self):
        ptcloud_xyz = self.points
        # print("The number of original points: " + str(ptcloud_xyz.shape))

        pcd = PointCloud()
        pcd.points = Vector3dVector(ptcloud_xyz)
        downpcd = voxel_down_sample(pcd, voxel_size=self.downcell_size)
        ptcloud_xyz_downed = np.asarray(downpcd.points)
        # print("The number of downsampled points: " + str(ptcloud_xyz_downed.shape))
        # draw_geometries([downpcd])

        if self.viz:
            draw_geometries([downpcd])
        sc = self.ptcloud2sc(ptcloud_xyz_downed, self.num_sector, self.num_ring, self.max_range)
        return sc
        # for res in range(len(ScanContext.sector_res)):
        #     num_sector = ScanContext.sector_res[res]
        #     num_ring = ScanContext.ring_res[res]
        #
        #     sc = self.ptcloud2sc(ptcloud_xyz_downed, num_sector, num_ring, self.max_range)
        #     self.SCs.append(sc)

    def plot_multiple_sc(self, fig_idx=1):

        num_res = len(ScanContext.sector_res)

        fig, axes = plt.subplots(nrows=num_res)

        axes[0].set_title('Scan Contexts with multiple resolutions', fontsize=14)
        for ax, res in zip(axes, range(num_res)):
            ax.imshow(self.SCs[res])

        plt.show()
