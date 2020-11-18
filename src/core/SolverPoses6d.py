
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
from core.ParticleSwarmOptimization import ParticleSwarmOptimization
from core.Conic import *
from core import Visualizer
from core import FileIO
from core import Conic



class SolverPoses6dDLT(object):
    def __init__(self, name_opt_method, **kwargs) -> None:
        avaliable_methods = {"Adam": Adam, "LM": LevenbergMarquardt}
        n_iters           = kwargs["n_iters"] if ("n_iters" in kwargs.keys()) else 10000
        alpha             = kwargs["alpha"]   if ("alpha"   in kwargs.keys()) else 1E-3
        beta1             = kwargs["beta1"]   if ("beta1"   in kwargs.keys()) else 0.9
        beta2             = kwargs["beta2"]   if ("beta2"   in kwargs.keys()) else 0.99
        if name_opt_method == "Adam":
            self.opt = avaliable_methods[name_opt_method](n_iters, alpha, beta1, beta2)
        if name_opt_method == "LM":
            self.opt = avaliable_methods[name_opt_method](n_iters, alpha)
            alpha = kwargs["alpha"] if ("alpha" in kwargs.keys()) else 1E-3
        self.pso = ParticleSwarmOptimization(100, 6, 500)
        return

    # def set_opt_hyper_params(self, **kwargs):
    def set_points3d_of_n_cams(self, points3d_of_n_cams: np.ndarray):
        self.points3d_of_n_cams = points3d_of_n_cams
        return

    def get_projection_mats_for_n_cams(self):
        self.mats_projections_of_n_cams = []
        for cam in self.cameras_pars:
            mat_intrin = np.array(cam["intrin"])
            mat_extrin = np.array(cam["extrin"])
            M = mat_intrin @ mat_extrin
            self.mats_projections_of_n_cams.append(M)
        return

    def set_cameras_pars(self, cameras_pars):
        self.cameras_pars = cameras_pars
        self.n_cameras = len(cameras_pars)
        self.get_projection_mats_for_n_cams()
        return

    def set_points2d_of_n_cams(self, points2d_of_n_cams):
        self.points2d_of_n_cams = points2d_of_n_cams
        return

    def run(self, theta0=np.zeros(6), **kwargs_fun_objective_multi):  
        self.opt.set_objective_func(geo.get_reprojection_error_multi)
        self.opt.set_jacobian_func(geo.get_jacobian_matrix)
        [log_loss, log_theta] = self.opt.run(theta0, **kwargs_fun_objective_multi)
        log = np.hstack((np.array(log_theta), np.array(log_loss).reshape((-1, 1))))
        return log


def loss_func_mixture_multi(rtvec: np.ndarray, args):
    Ms = args[0]
    points3d = args[1]
    points2d_obj_n_cams = args[2]


    n_cams = len(Ms)
    loss_total = 0
    for i_cam in range(n_cams):
        pt2d_head =points2d_obj_n_cams[i_cam][0]
        pt2d_pro = geo.project_points3d_to_2d(rtvec, Ms[i_cam], points3d[0:1, 0:3])

        pts2d_conic = points2d_obj_n_cams[i_cam][1:]
        pts3d_conic = points3d[1:]

        
        # DEBUG
        debug1 = FileIO.imread("C:/Users/Li/Desktop/Pose6dSolver-pyqt/姿态测量/images_solve/cam_1/scene_2.png")
        debug2 = FileIO.imread("C:/Users/Li/Desktop/Pose6dSolver-pyqt/姿态测量/images_solve/cam_2/scene_2.png")

        C = Ellipse2d()
        C._set_by_5points2d(pts2d_conic)
        C.draw([debug1, debug2][i_cam], thickness=2, color=(0, 255, 0))
        cv2.line([debug1, debug2][i_cam], Visualizer.to_plot(C.point2d_center), Visualizer.to_plot(pt2d_head), color=(0, 255, 0))
        
        E = Ellipse2d()
        E._set_by_5points2d(geo.project_points3d_to_2d(rtvec, Ms[i_cam], pts3d_conic))
        E.draw([debug1, debug2][i_cam], thickness=2, color=(255, 0, 0))
        cv2.line([debug1, debug2][i_cam], Visualizer.to_plot(pt2d_pro), Visualizer.to_plot(E.point2d_center), color=(255, 0, 0))
        cv2.namedWindow("DEBUG{}".format(i_cam+1), flags=cv2.WINDOW_FREERATIO)
        cv2.imshow("DEBUG{}".format(i_cam+1), [debug1, debug2][i_cam])
        cv2.waitKey(1)
        #print("index:", i_cam, "\tl1:\t",loss1, "\tl2:\t", loss2 )

        loss1 = ellipse_loss_func(rtvec, [Ms[i_cam], pts3d_conic, pts2d_conic])
        # sim   = np.dot(pt2d_pro, pt2d_head.T)[0]/(np.linalg.norm(pt2d_head) * np.linalg.norm(pt2d_pro)) 
        # loss2 = 1 - sim
        loss3 = np.linalg.norm(C.point2d_center - E.point2d_center)
        loss2 = np.linalg.norm(pt2d_head - pt2d_pro)
        loss_total += (loss1 + loss2 + loss3)
        
    return loss_total / n_cams


class SolverPoses6dConic(SolverPoses6dDLT):
    def __init__(self, name_opt_method, **kwargs):
        super().__init__(name_opt_method, **kwargs)
        self.pso = ParticleSwarmOptimization(64, 6, 1000)
        return

    def set_points3d(self, points3d: np.ndarray):
        super(SolverPoses6dConic, self).set_points3d_of_n_cams(points3d)
        self.points3d_for_conic = self.points3d_of_n_cams[1:]
        return
         
    def set_points2d_of_n_cams(self, points2d_of_n_cams):
        super().set_points2d_of_n_cams(points2d_of_n_cams)
        n_cams = len(self.points2d_of_n_cams)
        self.points2d_for_conic_of_n_cams = []
        for i_cam in range(n_cams):
            self.points2d_for_conic_of_n_cams.append(self.points2d_of_n_cams[i_cam][1:])
        return

    def run(self, theta0=np.zeros(6)):
        self.pso.set_objective_func(loss_func_mixture_multi)
        self.pso.set_boundery(
            lower_bound=np.array([-5, -5, -5, -np.pi, -np.pi, -np.pi]),
            upper_bound=np.array([+5, +5, +5, +np.pi, +np.pi, +np.pi])
        )
        if (theta0 == np.zeros(6)).all():
            self.pso.run(self.mats_projections_of_n_cams, self.points3d_of_n_cams, self.points2d_of_n_cams)
            theta0 = self.pso.global_best

        self.opt.set_objective_func(loss_func_mixture_multi)
        self.opt.set_jacobian_func(Conic.get_jacobian_matrix)

        #self.opt.set_residual_func(geo.get_residual)
        # theta0 = np.array([ 0.67959408, -0.53942265,  3.236483  , -0.13049411, -0.10265555, 0.01903561])
        [log_loss, log_theta] = self.opt.run(theta0, self.mats_projections_of_n_cams, self.points3d_of_n_cams, self.points2d_of_n_cams)
        log = np.hstack((np.array(log_theta), np.array(log_loss).reshape((-1, 1))))
        return log

def load_setttings(flags_points2d, settings):
    if   flags_points2d == 0:
        if   settings.FLAGS_OPT == 0:
            n_iters = settings.hyper_params_adam.n_iters
            alpha   = settings.hyper_params_adam.alpha
            beta1   = settings.hyper_params_adam.beta1
            beta2   = settings.hyper_params_adam.beta2
           
            solver = SolverPoses6dConic("Adam", n_iters=n_iters, alpha=alpha, beta1=beta1, beta2=beta2)
        elif settings.FLAG_OPT == 1:
            n_iters = settings.hyper_params_lm.n_iters
            alpha   = settings.hyper_params_lm.alpha
            solver = SolverPoses6dConic("LM", n_iters=n_iters, alpha=alpha)
    elif flags_points2d == 1:
        if   settings.FLAGS_OPT == 0:
            n_iters = settings.hyper_params_adam.n_iters
            alpha   = settings.hyper_params_adam.alpha
            beta1   = settings.hyper_params_adam.beta1
            beta2   = settings.hyper_params_adam.beta2
           
            solver = SolverPoses6dDLT("Adam", n_iters=n_iters, alpha=alpha, beta1=beta1, beta2=beta2)
        elif settings.FLAGS_OPT == 1:
            n_iters = settings.hyper_params_lm.n_iters
            alpha   = settings.hyper_params_lm.alpha
            solver = SolverPoses6dDLT("LM", n_iters=n_iters, alpha=alpha)
    return solver



