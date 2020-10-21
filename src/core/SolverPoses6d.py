
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt

sys.path.append("..")
from core import geometry as geo
from core.Adam import Adam
from core.LevenbergMarquardt import LevenbergMarquardt 
from core import Visualizer
from core import FileIO


class SolverPoses6d(object):
    def __init__(self, name_opt_method, **kwargs) -> None:
        avaliable_methods = {"Adam": Adam, "LM": LevenbergMarquardt}
        n_iters = kwargs["n_iters"] if ("n_iters" in kwargs.keys()) else 10000
        alpha = kwargs["alpha"] if ("alpha" in kwargs.keys()) else 1E-3
        beta1 = kwargs["beta1"] if ("beta1" in kwargs.keys()) else 0.9
        beta2 = kwargs["beta2"] if ("beta2" in kwargs.keys()) else 0.99
        if name_opt_method == "Adam":
            self.opt = avaliable_methods[name_opt_method](n_iters, alpha, beta1, beta2)
        if name_opt_method == "LM":
            # alpha = kwargs["alpha"] if ("alpha" in kwargs.keys()) else 1E-3
            # beta1 = kwargs["beta1"] if ("beta1" in kwargs.keys()) else 0.9
            # beta2 = kwargs["beta2"] if ("beta2" in kwargs.keys()) else 0.99
            self.opt = avaliable_methods[name_opt_method](n_iters, alpha)
            alpha = kwargs["alpha"] if ("alpha" in kwargs.keys()) else 1E-3

        return

    # def set_opt_hyper_params(self, **kwargs):


    def set_points3d(self, points3d: np.ndarray):
        self.points3d = points3d
        return

    def get_projection_mats_for_n_cams(self):
        self.mats_projections_for_n_cams = []
        for cam in self.cameras_pars:
            mat_intrin = np.array(cam["intrin"])
            mat_extrin = np.array(cam["extrin"])
            M = mat_intrin @ mat_extrin
            self.mats_projections_for_n_cams.append(M)
        return
    

    def set_cameras_pars(self, cameras_pars):
        self.cameras_pars = cameras_pars
        self.n_cameras = len(cameras_pars)
        self.get_projection_mats_for_n_cams()
        return

    def set_points2d_of_n_cams(self, points2d_of_n_cams):
        self.points2d_of_n_cams = points2d_of_n_cams
        return

    def run(self, theta0=np.zeros(6)):
        self.opt.set_objective_func(geo.get_reprojection_error_multi)
        self.opt.set_jacobian_func(geo.get_jacobian_matrix_multi)
        #self.opt.set_residual_func(geo.get_residual)
        n_points = self.points2d_of_n_cams[0].shape[0]
        [log_loss, log_theta] = self.opt.run(theta0, self.mats_projections_for_n_cams, self.points3d, self.points2d_of_n_cams)
        log = np.hstack((np.array(log_theta), np.array(log_loss).reshape((-1, 1))))
        return log

    def set_dir_output(self, dir_output: str="./"):
        self.dir_output = dir_output
        return




if __name__ == "__main__":
    from easydict import EasyDict
    dir_cam = "../../simu/results_calib/"
    dir_p3d = "../../simu/points3d_solve/"
    p3ds = np.loadtxt(dir_p3d+"/obj_{}.txt".format(0+1))
    cams = []
    p2ds = []
    for i in range(2):

        cam = FileIO.load_camera_pars(dir_cam+"/cam_{}/camera_pars.json".format(i+1))
        cams.append(cam)
    p2ds.append(np.array([
        [282,262],
        [653,659],
        [646,762], 
        [597,719],
        [709,696]
    ]))
    p2ds.append(np.array([
        [277,346],
        [956,692],
        [935,791], 
        [914,761],
        [976,710]
    ]))

    x = np.zeros(6)
    while True:
        img1 = np.zeros((1280, 1280, 3))
        img2 = np.zeros((1280, 1280, 3))
        vis = Visualizer.Visualizer()
        vis.draw_points2d(img1, p2ds[0])
        vis.draw_axis3d(img1, cams[0])
        vis.draw_backbone3d(img1, p3ds, x, cams[0])

        vis.draw_points2d(img2, p2ds[1])
        vis.draw_axis3d(img2, cams[1])
        vis.draw_backbone3d(img2, p3ds, x, cams[1])
        cv2.namedWindow("1", cv2.WINDOW_FREERATIO)
        cv2.namedWindow("2", cv2.WINDOW_FREERATIO)
        cv2.imshow("1", img1)
        cv2.imshow("2", img2)
        key = cv2.waitKey(500)
        if key == ord("q"):
            x[0] += 0.01
        if key == ord("w"):
            x[0] -= 0.01
        if key == ord("a"):
            x[1] += 0.01
        if key == ord("s"):
            x[1] -= 0.01
        if key == ord("z"):
            x[2] += 0.01
        if key == ord("x"):
            x[2] -= 0.01

        if key == ord("r"):
            x[3] += 0.01
        if key == ord("t"):
            x[3] -= 0.01
        if key == ord("f"):
            x[4] += 0.01
        if key == ord("g"):
            x[4] -= 0.01
        if key == ord("v"):
            x[5] += 0.01
        if key == ord("b"):
            x[5] -= 0.01
        if key == ord("p"):
            break


    solver = SolverPoses6d("LM", n_iters=1000, alpha=1E-2)
    solver.set_cameras_pars(cams)
    solver.set_points2d_of_n_cams(p2ds)
    solver.set_points3d(p3ds)
    solver.run(x)
    vis = Visualizer.Visualizer()
    vis.draw_points2d(img1, p2ds[0])
    vis.draw_axis3d(img1, cams[0])
    vis.draw_backbone3d(img1, p3ds, solver.opt.theta, cams[0])

    vis.draw_points2d(img2, p2ds[1])
    vis.draw_axis3d(img2, cams[1])
    vis.draw_backbone3d(img2, p3ds, solver.opt.theta, cams[1])
    while 1:
        cv2.imshow("1", img1)
        cv2.imshow("2", img2)
        cv2.waitKey(100)
    
    print()


