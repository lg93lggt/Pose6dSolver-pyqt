

import argparse
import os
from typing import Tuple

import cv2
import numpy as np
import sys

sys.path.append("..")
from core import geometry as geo


class CalibratorByDLT(object):
    def __init__(self, n_points=8, unit_length_meter=1) -> None:
        self.unit_length_meter = unit_length_meter
        self.n_points = n_points
        self.solve_perspective_mat_3d_to_2d = geo.solve_projection_mat_3d_to_2d
        # RQ decompose or normal
        self.decompose_intrin_extrin_from_projection_mat = geo.decompose_projection_mat
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
        M = self.solve_perspective_mat_3d_to_2d(self.points3d_real, self.points2d_obj, method="ols")
        [mat_intrin, mat_extrin] = self.decompose_intrin_extrin_from_projection_mat(M)
 
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


