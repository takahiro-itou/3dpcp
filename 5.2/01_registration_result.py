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
