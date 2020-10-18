

import argparse
import os
from typing import Tuple

import cv2
import numpy as np

from . import geometry as geo
from . import FileIO 
from .  Visualizer import Visualizer


class CalibratorByDLT(object):
    def __init__(self, n_points=8, unit_length_meter=1) -> None:
        self.unit_length_meter = unit_length_meter
        self.n_points = n_points
        if   self.n_points == 8:
            self.points3d_unit_cube = np.array(
                [[0, 0, 0],
                 [1, 0, 0],
                 [1, 1, 0],
                 [0, 1, 0],
                 [0, 0, 1],
                 [1, 0, 1],
                 [1, 1, 1],
                 [0, 1, 1]],
                dtype=np.float
            )
        elif self.n_points == 7:
            self.points3d_unit_cube = np.array(
                [[0, 0, 0],
                 [1, 0, 0],
                 [1, 1, 0],
                #[0, 1, 0],
                 [0, 0, 1],
                 [1, 0, 1],
                 [1, 1, 1],
                 [0, 1, 1]],
                dtype=np.float
            )
        elif self.n_points == 6:
            self.points3d_unit_cube = np.array(
                [[0, 0, 0],
                 [1, 0, 0],
                #[1, 1, 0],
                #[0, 1, 0],
                 [0, 0, 1],
                 [1, 0, 1],
                 [1, 1, 1],
                 [0, 1, 1]],
                dtype=np.float
            )
        self.points3d_real = self.points3d_unit_cube * self.unit_length_meter
        
        self.solve_perspective_mat_3d_to_2d = geo.solve_projection_mat_3d_to_2d
        self.decomposition_intrin_extrin_from_projection_mat = geo.decomposition_intrin_extrin_from_projection_mat
        self.R2r = geo.R_to_r
        self.T2t = geo.T_to_t
        return

    def set_points3d(self, points3d: np.ndarray):
        self.points3d_real = points3d 
        return

    def set_points2d(self, points2d: np.ndarray):
        self.points2d_obj = points2d
        return
    
    def solve(self):
        M = self.solve_perspective_mat_3d_to_2d(self.points3d_real, self.points2d_obj)

        [mat_intrin, mat_extrin] = self.decomposition_intrin_extrin_from_projection_mat(M)
 
        rvec = self.R2r(mat_extrin)
        tvec = self.T2t(mat_extrin)
        self.camera_pars = {}
        self.camera_pars["intrin"] = mat_intrin
        self.camera_pars["extrin"] = mat_extrin
        self.camera_pars["rvec"] = rvec
        self.camera_pars["tvec"] = tvec
        return

    def outprint(self):
        print()
        print("intrin:\n", self.camera_pars["intrin"])
        print("extrin:\n", self.camera_pars["extrin"])
        print("rvec:\n", self.camera_pars["rvec"])
        print("tvec:\n", self.camera_pars["tvec"])
        print()
        return
    
    def run(self):
        print("\n开始标定...")
        self.solve()
        self.outprint()
        return

def main(args_cmd, **k_args):
    mode = "calib"

    vis = Visualizer()

    n_points = args_cmd[0]
    unit_length = args_cmd[1]
    fio = FileIO.FileIO()
    fio  = k_args["fio"]

    n_cams   = fio.file_structure[mode]["n_cams"]
    n_senses = fio.file_structure[mode]["n_senses"]
    dir_points2d  = fio.file_structure[mode]["dirs"]["points2d"]
    dir_images    = fio.file_structure[mode]["dirs"]["images"]
    dir_results   = fio.file_structure[mode]["dirs"]["results"]
    dir_visualize = fio.file_structure[mode]["dirs"]["visualize"]
    names_subdir = fio.file_structure[mode]["names_subdir"]
    suffix_image = fio.file_structure[mode]["suffix_image"]
    for i_sense in range(n_senses):
        print("sense:\t{} / {}".format(i_sense + 1, n_senses))
        pair = fio.file_structure[mode]["pairs"][i_sense]

        for i_cam in range(n_cams):
            dir_points2d = fio.file_structure[mode]["dirs"]["points2d"]
            pth_points2d = os.path.join(dir_points2d, names_subdir[i_cam], pair[i_cam] + ".txt")

            calibrator = CalibratorByDLT(n_points, unit_length)
            points2d = fio.load_points2d(pth_points2d)
            calibrator.set_points2d(points2d)
            calibrator.run()
            dir_camera_par_output = os.path.join(dir_results, names_subdir[i_cam])
            fio.save_camera_pars(dir_camera_par_output, calibrator.camera_pars)

            pth_image = os.path.join(dir_images, names_subdir[i_cam], pair[i_cam] + suffix_image)
            img = cv2.imread(pth_image)
            vis.draw(img=img, mode="calib", points2d=points2d, points3d=calibrator.points3d_real, camera_pars=calibrator.camera_pars)
            cv2.imshow("cam_{}".format(i_cam + 1), img)
            cv2.waitKey(100)

            dir_image_output = os.path.join(dir_visualize, names_subdir[i_cam])
            fio.save_image(dir_image=dir_image_output, prefix=FileIO.split_path(pth_image)[1], img=img)
    return
    

