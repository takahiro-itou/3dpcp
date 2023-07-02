#!/usr/bin/python3
##  -*-  coding: utf-8-unix; mode: python  -*-  ##

import os

dirname = 'rgbd-dataset'
classes = [ 'apple', 'banana', 'camera' ]

url = 'https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_pcd_ascii'

for cls_name in classes:
    if not os.path.exists("{0}/{1}".format(dirname, cls_name,)):
        os.system("wget {0}/{1}_1.tar".format(url, cls_name,))
        os.system("tar -xvf {0}_1.tar".format(cls_name,))
    # End If
# Next (i)
