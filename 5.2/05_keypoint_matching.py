#!/usr/bin/python3
##  -*-  coding: utf-8-unix; mode: python  -*-  ##

import copy
import numpy
import open3d
import sys

def create_lineset_from_correspondences(
        corrs_set, pcd1, pcd2, transformation=numpy.identity(4)):

    pcd1_temp = copy.deepcopy(pcd1)
    pcd1_temp.transform(transformation)
    corrs = numpy.asarray(corrs_set)

    np_points1 = numpy.array(pcd1_temp.points)
    np_points2 = numpy.array(pcd2.points)
    points = list()
    lines  = list()

    for i in range(corrs.shape[0]):
        points.append( np_points1[corrs[i, 0]] )
        points.append( np_points2[corrs[i, 1]] )
        lines.append( [2*i, (2*i) + 1] )
    # Next (i)

    colors = [ numpy.random.rand(3) for i in range(len(lines)) ]
    line_set = open3d.geometry.LineSet(
        points = open3d.utility.Vector3dVector(points),
        lines  = open3d.utility.Vector2iVector(lines),
    )
    line_set.colors = open3d.utility.Vector3dVector(colors)
    return  line_set
# End Def (create_lineset_from_correspondences)


def draw_registration_result(
        source, target, transformation,
        front  = [ 0.024, -0.025, -0.973 ],
        lookat = [ 0.488,  1.722,  1.556 ],
        up     = [ 0.047, -0.972,  0.226 ]):

    pcds = list()
    for s in source:
        temp = copy.deepcopy(s)
        pcds.append(temp.transform(transformation))
    pcds += target

    open3d.visualization.draw_geometries(
        pcds, zoom=0.3199, front=front, lookat=lookat, up=up,)

# End Def (draw_registration_result)

def keypoint_and_feature_extraction(pcd, voxel_size):

    keypoints = pcd.voxel_down_sample(voxel_size)

    viewpoint = numpy.array([ 0.0, 0.0, 0.0 ], dtype='float64')
    radius_normal = 2.0 * voxel_size

    keypoints.estimate_normals(
        open3d.geometry.KDTreeSearchParamHybrid(
            radius = radius_normal,max_nn=30))
    keypoints.orient_normals_towards_camera_location(viewpoint)

    radius_feature = 5.0 * voxel_size
    feature = open3d.pipelines.registration.compute_fpfh_feature(
        keypoints,
        open3d.geometry.KDTreeSearchParamHybrid(
            radius = radius_feature, max_nn=100))

    return (keypoints, feature)
# End Def (keypoint_and_feature_extraction

if __name__ == '__main__':

    # データ読み込み
    filename_source = sys.argv[1]
    filename_target = sys.argv[2]
    source = open3d.io.read_point_cloud(filename_source)
    target = open3d.io.read_point_cloud(filename_target)

    source.paint_uniform_color([0.5, 0.5, 1.0])
    target.paint_uniform_color([1.0, 0.5, 0.5])
    initial_trans = numpy.identity(4)
    initial_trans[0, 3] = -3.0

    # draw_registration_result([source], [target], initial_trans)

    # 特徴量抽出
    voxel_size = 0.1
    s_kp, s_feature = keypoint_and_feature_extraction(source, voxel_size)
    t_kp, t_feature = keypoint_and_feature_extraction(target, voxel_size)

    s_kp.paint_uniform_color([0.0, 1.0, 0.0])
    t_kp.paint_uniform_color([0.0, 1.0, 0.0])
    # draw_registration_result([s_kp], [t_kp], initial_trans)
    # draw_registration_result([source, s_kp], [target, t_kp], initial_trans)

    # 対応点探索
    np_s_feature = s_feature.data.T
    np_t_feature = t_feature.data.T
    corrs = open3d.utility.Vector2iVector()
    threshold = 0.9
    for i, feat in enumerate(np_s_feature):
        distance = numpy.linalg.norm(np_t_feature - feat, axis = 1)
        nearest_idx = numpy.argmin(distance)
        dist_order = numpy.argsort(distance)
        ratio = distance[dist_order[0]] / distance[dist_order[1]]
        if ratio < threshold:
            corr = numpy.array( [[i], [nearest_idx]], numpy.int32 )
            corrs.append(corr)
        # End If (ratio)
    # Next (i, feat)
    print("対応点セットの数 : {0}".format(len(corrs),), file=sys.stdout)

    # 姿勢計算
    # 全点利用
    print('全点利用による姿勢計算')
    line_set = create_lineset_from_correspondences(
        corrs, s_kp, t_kp, initial_trans)
    draw_registration_result(
        [source, s_kp], [target, t_kp, line_set], initial_trans)

    trans_ptp = open3d.pipelines.registration.TransformationEstimationPointToPoint(False)
    trans_all = trans_ptp.compute_transformation(s_kp, t_kp, corrs)
    print(trans_all)
    # draw_registration_result([source], [target], trans_all)

    # RANSAC利用
    print('RANSACによる姿勢計算')
    distance_threshold = voxel_size * 0.5
    result = open3d.pipelines.registration.registration_ransac_based_on_correspondence(
        s_kp, t_kp, corrs, distance_threshold,
        open3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n = 3,
        checkers = [
            open3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            open3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria = open3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )

    line_set = create_lineset_from_correspondences(
        result.correspondence_set, s_kp, t_kp, initial_trans)
    draw_registration_result(
        [source, s_kp], [target, t_kp, line_set], initial_trans)

    print(result.transformation)
    draw_registration_result([source], [target], result.transformation)
