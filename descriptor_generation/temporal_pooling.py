""" Temporal segment correspondence estimation and feature pooling. """

import numpy as np
from sklearn.neighbors import KDTree
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.misc_utils import *
from utils.kitti_dataloader import *

# 获得在之前段落中的对应点云段
def get_segment_correspondences(idx, database_dict):
    """ Return the corresponding segment IDs from previous frame """
    corres_segment = []

    # Return nans if previous frame as low number of segments.
    # 如果特征数少于一定的阈值，则认为是无效点云段
    if len(database_dict['features_database'][idx-1]) < 7:
        for s in range(len(database_dict['segments_database'][idx])):
            corres_segment.append(np.nan)
        return corres_segment

    # Calculate segment centroids of previous frame.
    # 计算之前点云段的质心
    prev_centroids = []
    for ps in database_dict['segments_database'][idx-1]:
        prev_centroids.append(np.mean(ps, axis=0))

    segments = database_dict['segments_database'][idx]
    features = database_dict['features_database'][idx]
    # 记录相邻帧的位姿变换
    rel_T = database_dict['rel_transforms'][idx-1]

    # KDTrees for feature-space and Euclidean-space
    # 创建上一帧数据的点云段特征kdtree和质心kdtree
    ftree = KDTree(database_dict['features_database'][idx-1])
    ctree = KDTree(np.asarray(prev_centroids))

    for s in range(len(segments)):
        # Cordinates of current segment centroid wrt previous frame.
        centroid = np.mean(segments[s], axis=0)
        # 最后一位加上1，从欧式坐标转换为齐次坐标，便于之后和变换矩阵相乗获得新的点云段质心
        rel_centroid = euclidean_to_homogeneous(centroid)
        rel_centroid = np.matmul(rel_T, rel_centroid)
        # 再转回正常的欧式坐标
        centroid_new = homogeneous_to_euclidean(rel_centroid)

        # Feature-space NNs
        # 从当前帧的点云段和质心中寻找与上一帧的最近邻，发现对应关系
        # 点云的段特征KDtree，找5个最近邻 distf与查询点的距离，indf对应的特征序号
        distf, indf = ftree.query(features[s].reshape(1, -1), k=5)
        # Euclidean-space NNs
        # 欧式空间的KDtree，找到距离r=2内的最近邻数。 indc：对应点的索引， distc对应点的距离
        indc, distc = ctree.query_radius(centroid_new.reshape(
            1, -1), 2, return_distance=True, count_only=False, sort_results=True)
        # Correspondence candidates
        # 计算点云段最近邻和质心最近邻的交集，确定对应关系
        ind_common = np.intersect1d(indf, indc[0])
        if len(ind_common) < 1:
            # Return nan if zero candidates
            corres_segment.append(np.nan)
            continue
        elif len(ind_common) > 1:
            # If more than one candidate,
            min_ind_common = ind_common[0]
            # 选出交集中的最小距离
            min_cdist = distc[0][np.where(indc[0] == ind_common[0])]
            minf_dist = distf[np.where(indf == ind_common[0])]
            # Find the ID of segment which minimizes both feature-space and Euclidean-space distance
            # 迭代找出特征空间和欧式空间中的距离最近的点
            for ind_com in range(1, len(ind_common)):
                cdist_check = distc[0][np.where(
                    indc[0] == ind_common[ind_com])] - min_cdist
                fdist_check = distf[np.where(
                    indf == ind_common[ind_com])] - minf_dist
                if cdist_check < 0 and fdist_check < 0:
                    min_ind_common = ind_common[ind_com]
                    min_cdist = distc[0][np.where(
                        indc[0] == ind_common[ind_com])]
                    minf_dist = distf[np.where(indf == ind_common[ind_com])]
        else:
            min_ind_common = ind_common[0]
            min_cdist = distc[0][np.where(indc[0] == ind_common[0])]
            minf_dist = distf[np.where(indf == ind_common[0])]
        corres_segment.append([min_ind_common, min_cdist[0], minf_dist[0]])
    return corres_segment


# 
def get_temporal_features(idx, n_frames_max, n_count, database_dict):
    """ Return the pooled feature using all temporal correspondences """
    features = database_dict['features_database'][idx]

    # Return nan if low number of segments.
    if len(features) < 7:
        database_dict['seg_corres_database'].append([])
        return []

    if(idx > 0):
        database_dict['seg_corres_database'].append(
            get_segment_correspondences(idx, database_dict))

    pooled_softmax_features = np.zeros((len(features), np.shape(features)[1]))
    n_frames = min(idx, n_frames_max)

    # Segment-wise feature pooling
    for s in range(len(features)):
        pooled_softmax_features[s] += features[s]
        if n_frames < 1:  # No correspondences for 0th frame
            continue

        seg_ind = s
        past_features_database = []
        past_features_dist = []

        # Get features of all previous correspondences
        for n in range(1, n_frames + 1):
            past_features = database_dict['features_database'][idx - n]
            if len(database_dict['seg_corres_database'][idx - n]) == 0:
                break
            segment_corres = database_dict['seg_corres_database'][idx - n][seg_ind]
            if is_nan(segment_corres):
                break
            seg_ind = segment_corres[0]
            past_features_database.append(past_features[seg_ind])
            past_features_dist.append(np.linalg.norm(
                features[s] - past_features[seg_ind]))

        # Calculate pooling weights
        exp_dists = np.exp(-0.1*np.asarray(past_features_dist))
        n_count.append(len(past_features_dist))
        exp_dists /= np.sum(exp_dists)

        # Pool features of all previous correspondences
        for p in range(len(past_features_database)):
            pooled_softmax_features[s] += exp_dists[p] * \
                past_features_database[p]
    return pooled_softmax_features

#####################################################################################
# Test
#####################################################################################


if __name__ == "__main__":

    seq = '06'
    base_dir = '/mnt/7a46b84a-7d34-49f2-b8f0-00022755f514/'
    n_frames_max = 3

    poses_file = base_dir + 'datasets/Kitti/dataset/sequences/' + seq + '/poses.txt'
    transforms, _ = load_poses_from_txt(poses_file)
    rel_transforms = get_delta_pose(transforms)

    data_dir = base_dir + 'seg_test/kitti/' + seq
    features_database = load_pickle(data_dir + '/features_database.pickle')
    segments_database = load_pickle(data_dir + '/segments_database.pickle')

    num_queries = len(features_database)
    seg_corres_database = []
    database_dict = {'segments_database': segments_database,
                     'features_database': features_database,
                     'seg_corres_database': seg_corres_database,
                     'rel_transforms': rel_transforms}
    pooled_softmax_features_database = []
    n_count = []

    for query_idx in range(num_queries):
        pooled_softmax_features = get_temporal_features(
            query_idx, n_frames_max, n_count, database_dict)
        pooled_softmax_features_database.append(pooled_softmax_features)

        if (query_idx % 100 == 0):
            print('', query_idx, 'complete:', (query_idx*100)/num_queries, '%')
            sys.stdout.flush()

    save_dir = '/mnt/bracewell/seg_test/kitti/' + seq
    save_pickle(pooled_softmax_features_database, save_dir +
                '/second_order/temporal_features_database.pickle')

    print('')
    print('Avg frames aggregated: ', np.mean(n_count))

