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
            radius=0.03, max_nn=100)
        )
    sum_fpfh = numpy.sum(numpy.array(fpfh.data), 1)
    return ( sum_fpfh / numpy.linalg.norm(sum_fpfh) )
# End Def (extract_fpfh)

dirname = "rgbd-dataset"
classes = ["apple", "banana", "camera"]

num_samples = 100
feat_train = numpy.zeros( (len(classes), num_samples, 33) )
feat_test  = numpy.zeros( (len(classes), num_samples, 33) )

for i, cls_name in enumerate(classes):
    print("Extracting train features in {0} ...".format(cls_name,))
    for n in range(num_samples):
        filename = "{0}/{1}/{1}_1/{1}_1_1_{2}.pcd".format(
            dirname, cls_name, n + 1,)
        feat_train[i, n] = extract_fpfh(filename)
    # Next (n)

    print("Extracting test features in {0} ...".format(cls_name,))
    for n in range(num_samples):
        filename = "{0}/{1}/{1}_1/{1}_1_4_{2}.pcd".format(
            dirname, cls_name, n + 1,)
        feat_test[i, n] = extract_fpfh(filename)
    # Next (n)
# Next (i, cls_name)
