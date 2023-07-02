#!/usr/bin/python3
##  -*-  coding: utf-8-unix; mode: python  -*-  ##

import copy
import numpy
import open3d
import sys

def draw_registration_result(source, target, transformation):

    pcds = list()
    for s in source:
        temp = copy.deepcopy(s)
        pcds.append(temp.transform(transformation))
    pcds += target

    open3d.visualization.draw_geometries(
        pcds, zoom=0.3199,
        front  = [ 0.024, -0.025, -0.973 ],
        lookat = [ 0.488,  1.722,  1.556 ],
        up     = [ 0.047, -0.972,  0.226 ]
    )

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

    draw_registration_result([source], [target], initial_trans)

    # 特徴量抽出
    voxel_size = 0.1
    s_kp, s_feature = keypoint_and_feature_extraction(source, voxel_size)
    t_kp, t_feature = keypoint_and_feature_extraction(target, voxel_size)

    s_kp.paint_uniform_color([0.0, 1.0, 0.0])
    t_kp.paint_uniform_color([0.0, 1.0, 0.0])
    draw_registration_result([s_kp], [t_kp], initial_trans)
    draw_registration_result([source, s_kp], [target, t_kp], initial_trans)
