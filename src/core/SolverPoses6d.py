
import argparse
import json
import math
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

from . import FileIO
from . import geometry as geo
from .Adam import Adam
from .Visualizer import Visualizer


class SolverPoses6d(object):
    def __init__(self) -> None:
        self.opt = Adam(n_iters=10000, alpha=1E-3, beta1=0.9, beta2=0.99)
        #self.opt2 = Adam(n_iters=50000, alpha=1E-6, beta1=0.9, beta2=0.999)
        return

    def set_points3d(self, points3d: np.ndarray):
        self.points3d = points3d
        return

    def get_projection_mats_for_n_cams(self):
        self.mats_projections_for_n_cams = []
        for cam in self.cameras_pars:
            mat_intrin = cam["intrin"]
            mat_extrin = cam["extrin"]
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
        n_points = self.points2d_of_n_cams[0].shape[0]
        [log_loss, log_theta] = self.opt.run(theta0, self.mats_projections_for_n_cams, self.points3d[:n_points], self.points2d_of_n_cams)
        log = np.hstack((np.array(log_theta), np.array(log_loss).reshape((-1, 1))))
        return log

    def set_dir_output(self, dir_output: str="./"):
        self.dir_output = dir_output
        return




def main(args, **k_args):
    mode = "solve"

    vis = Visualizer()

    fio = FileIO.FileIO()
    fio = k_args["fio"]

    pth_points3d = args.load_model_points3d[0]
    points3d = fio.load_points3d(pth_points3d)
    

    n_cams   = fio.file_structure[mode]["n_cams"]
    n_senses = fio.file_structure[mode]["n_senses"]
    dir_points2d  = fio.file_structure[mode]["dirs"]["points2d"]
    dir_cameras   = fio.file_structure["calib"]["dirs"]["results"]
    dir_images    = fio.file_structure[mode]["dirs"]["images"]
    dir_logs      = fio.file_structure[mode]["dirs"]["logs"]
    dir_results   = fio.file_structure[mode]["dirs"]["results"]
    dir_visualize = fio.file_structure[mode]["dirs"]["visualize"]
    names_subdir = fio.file_structure[mode]["names_subdir"]
    suffix_image = fio.file_structure[mode]["suffix_image"]
    cameras_pars = []
    for i_cam in range(n_cams):
        name_subdir = names_subdir[i_cam]
        pth_cameras = os.path.join(dir_cameras, name_subdir, "camera_pars.json")
        camera_pars = fio.load_camera_pars(pth_cameras)
        cameras_pars.append(camera_pars)

    for i_sense in range(n_senses):
        print("sense:\t{:d} / {:d}".format(i_sense + 1, n_senses))
        pair = fio.file_structure[mode]["pairs"][i_sense]
        points2d_of_n_cams = []
        for i_cam in range(n_cams):
            name_subdir = names_subdir[i_cam]
            prefix_points2d = pair[i_cam]
            pth_points2d = os.path.join(dir_points2d, name_subdir, prefix_points2d + ".txt")
            points2d = fio.load_points2d(pth_points2d)
            points2d_of_n_cams.append(points2d)

        solver = SolverPoses6d()
        solver.set_cameras_pars(cameras_pars)
        solver.set_points2d_of_n_cams(points2d_of_n_cams)    
        solver.set_points3d(points3d)   
        res = solver.run()
        
        n_iters = res.shape[0]
        x = np.arange(n_iters)
        plt.plot(x, res[:, 0])
        plt.draw()
        
        fio.save_log(dir_logs=dir_logs, prefix=str(i_sense), log=res)
        fio.save_theta(dir_logs=dir_results, prefix=str(i_sense), log=solver.opt.theta)
        degree = solver.opt.theta[:3] / np.pi * 180
        print("theta=", solver.opt.theta)
        print("degree=", degree)
        print()

        for i_cam in range(n_cams):
            pth_image = os.path.join(dir_images, names_subdir[i_cam], pair[i_cam] + suffix_image)
            subdir_visualize = os.path.join(dir_visualize, names_subdir[i_cam])

            img = cv2.imread(pth_image)
            vis.draw(
                mode=mode,
                img=img, 
                points2d=points2d_of_n_cams[i_cam], 
                points3d=points3d, 
                rtvec=solver.opt.theta, 
                camera_pars=cameras_pars[i_cam]
            )
            cv2.imshow("cam_" + str(i_cam + 1), img)
            cv2.waitKey(100)
            fio.save_image(
                dir_image=subdir_visualize, 
                prefix=pair[i_cam], 
                img=img
            ) 
            if args.load_model3d:
                pth_model = args.load_model3d[0]
                model = fio.load_model_from_stl_binary(pth_model)
                vis.draw_model3d(img, solver.opt.theta, cameras_pars[i_cam], model)
                cv2.imshow(str(i_cam), img)
                cv2.waitKey(100)
            fio.save_image(
                dir_image=subdir_visualize, 
                prefix=pair[i_cam] + "_with_model", 
                img=img
            ) 
            
    return




