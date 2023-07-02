#!/usr/bin/python3
##  -*-  coding: utf-8-unix; mode: python  -*-  ##

import open3d
import numpy

def extract_fpfh(filename):

    print(" ", filename)
    pcd = open3d.io.read_point_cloud(filename)
    pcd = pcd.voxel_down_sample(0.01)
    pcd.estimate_normals(
        search_param = open3d.geometry.KDTreeSearchParamHybrid(
            radius=0.02, max_nn=10)
        )
    fpfh = open3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        search_param = open3d.geometry.KDTreeSearchParamHybrid(
            radius=0.03, max_nn-100)
        )
    sum_fpfh = numpy.sum(numpy.array(fpfh.data), 1)
    return ( sum_fpfh / numpy.linalg.norm(sum_fpfh) )
# End Def (extract_fpfh)
