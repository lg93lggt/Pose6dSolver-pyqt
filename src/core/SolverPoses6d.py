
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt

from core import geometry


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
            # alpha = kwargs["alpha"] if ("alpha" in kwargs.keys()) else 1E-3
            # beta1 = kwargs["beta1"] if ("beta1" in kwargs.keys()) else 0.9
            # beta2 = kwargs["beta2"] if ("beta2" in kwargs.keys()) else 0.99
            self.opt = avaliable_methods[name_opt_method](n_iters, alpha)
            alpha = kwargs["alpha"] if ("alpha" in kwargs.keys()) else 1E-3
        self.pso = ParticleSwarmOptimization(100, 6, 500)
        return

    # def set_opt_hyper_params(self, **kwargs):
    def set_points3d(self, points3d: np.ndarray):
        self.points3d_of_n_cams = points3d
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
        self.pso.set_objective_func(geo.get_reprojection_error_multi)
        self.pso.set_boundery(
            lower_bound=np.array([-np.pi, -np.pi, -np.pi, -5, -5, -5]),
            upper_bound=np.array([+np.pi, +np.pi, +np.pi, +5, +5, +5])
        )
  
        self.opt.set_objective_func(geo.get_reprojection_error_multi)
        self.opt.set_jacobian_func(geo.get_jacobian_matrix_multi)
        #self.opt.set_residual_func(geo.get_residual)
        if (theta0 == np.zeros(6)).all():
            self.pso.run(self.mats_projections_for_n_cams, self.points3d_of_n_cams[0], self.points2d_of_n_cams)
            theta0 = self.pso.global_best

        # n_points = self.points2d_of_n_cams[0].shape[0]
        [log_loss, log_theta] = self.opt.run(theta0, self.mats_projections_for_n_cams, self.points3d_of_n_cams[0], self.points2d_of_n_cams)
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
        pt2d_pro = geometry.project_points3d_to_2d(rtvec, Ms[i_cam], points3d[0:1, 0:3])

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
        super(SolverPoses6dConic, self).set_points3d(points3d)
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
            self.pso.run(self.mats_projections_for_n_cams, self.points3d_of_n_cams, self.points2d_of_n_cams)
            theta0 = self.pso.global_best

        self.opt.set_objective_func(loss_func_mixture_multi)
        self.opt.set_jacobian_func(Conic.get_jacobian_matrix)

        #self.opt.set_residual_func(geo.get_residual)
        # theta0 = np.array([ 0.67959408, -0.53942265,  3.236483  , -0.13049411, -0.10265555, 0.01903561])
        [log_loss, log_theta] = self.opt.run(theta0, self.mats_projections_for_n_cams, self.points3d_of_n_cams, self.points2d_of_n_cams)
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


if __name__ == "__main__":
    vis = Visualizer.Visualizer()
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

    retval, rvecs, tvecs	=	cv2.solveP3P(p3ds[:3], p2ds[0][:3].astype(np.float), cams[0]["intrin"][:3, :3], np.zeros(4), cv2.SOLVEPNP_P3P)



    x = np.array([ 0.21, -2.89, -1.12,  0.72, -0.04,  0.07])
    while True:
        img1 = np.zeros((1280, 1280, 3))
        img2 = np.zeros((1280, 1280, 3))
        #break
        vis.draw_points2d_with_texts(img1, p2ds[0])
        vis.draw_axis3d(img1, cams[0])
        vis.draw_points3d_with_texts(img1, p3ds, x, cams[0])

        vis.draw_points2d_with_texts(img2, p2ds[1])
        vis.draw_axis3d(img2, cams[1])
        vis.draw_points3d_with_texts(img2, p3ds, x, cams[1])
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
            x[3] += 0.1
        if key == ord("t"):
            x[3] -= 0.1
        if key == ord("f"):
            x[4] += 0.1
        if key == ord("g"):
            x[4] -= 0.1
        if key == ord("v"):
            x[5] += 0.1
        if key == ord("b"):
            x[5] -= 0.1
        if key == ord("p"):
            break


    solver = SolverPoses6d("Adam", n_iters=10000, alpha=1E-3, beta1=0.9, beta2=0.999)#("LM", n_iters=1000, alpha=1E-3)
    solver.set_cameras_pars(cams)
    solver.set_points2d_of_n_cams(p2ds)
    solver.set_points3d(p3ds)
    # solver.run(np.zeros(6))
    # 从0优化: loss: 97.962422, array([ 0.35450463, -1.48063947, -0.71277299,  0.16836641,  0.2146703 , -0.3727009 ])
    solver.opt.theta =  np.array([ 0.35450463, -1.48063947, -0.71277299,  0.16836641,  0.2146703 , -0.3727009 ])

    pso = ParticleSwarmOptimization(50, 6, 1000)
    pso.set_objective_func(geo.get_reprojection_error_multi)
    pso.set_boundery(
        lower_bound=np.array([-5, -5, -5, -np.pi, -np.pi, -np.pi]),
        upper_bound=np.array([+5, +5, +5, +np.pi, +np.pi, +np.pi])
    )
    # pso.run(solver.mats_projections_for_n_cams, p3ds, p2ds)
    # y = pso.global_best
    # PSO从0优化: loss: 173,

    solver2 = SolverPoses6d("Adam", n_iters=10000, alpha=1E-3, beta1=0.9, beta2=0.999)
    solver2.set_cameras_pars(cams)
    solver2.set_points2d_of_n_cams(p2ds)
    solver2.set_points3d(p3ds)
    # solver2.run(y)
    # 接PSO优化结果: loss: 11.45, array([-0.2024998 ,  2.92787609,  1.2343545 ,  0.72375618, -0.03668649,  0.07028912])
    solver2.opt.theta =  np.array([-0.2024998 ,  2.92787609,  1.2343545 ,  0.72375618, -0.03668649,  0.07028912])

    img1 = np.zeros((1280, 1280, 3))
    img2 = np.zeros((1280, 1280, 3))
    vis = Visualizer.Visualizer()
    vis.draw_points2d_with_texts(img1, p2ds[0])
    vis.draw_axis3d(img1, cams[0])
    vis.draw_points3d_with_texts(img1, p3ds, solver.opt.theta, cams[0])
    vis.draw_backbone3d(img1, p3ds, solver2.opt.theta, cams[0])

    vis.draw_points2d_with_texts(img2, p2ds[1])
    vis.draw_axis3d(img2, cams[1])
    vis.draw_points3d_with_texts(img2, p3ds, solver.opt.theta, cams[1])
    vis.draw_backbone3d(img2, p3ds, solver2.opt.theta, cams[1])
    while 1:
        cv2.imshow("1", img1)
        cv2.imshow("2", img2)
        cv2.waitKey(100)
    
    print()


