
import os
import sys
from math import nan
from typing import Any, Dict, List

import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

sys.path.append("..")
from core  import Ellipse2d

from . import FileIO, eps
from . import geometry as geo
from .Adam import Adam
from . import Visualizer


class Conic2d(object):
    """
    Conic(*args): \n
        1: init by 6 pars \n
        ax^2 + bxy + cy^2 + dx + ey + f = 0

        2: init by mat C \n
        x.T * C * x = 0
    """
    def __init__(self, *args) -> None:
        if args is None:
            self.mat = np.zeros((3, 3))
        elif len(args) == 1:
            if type(args[0]) is np.ndarray:
                input_mat = args[0]
                self._set_by_mat(mat_input=input_mat)
        return

    def __refresh(self):
        self.a = self.mat[0, 0]
        self.b = self.mat[0, 1] * 2
        self.c = self.mat[1, 1]
        self.d = self.mat[2, 0] * 2
        self.e = self.mat[2, 1] * 2
        self.f = self.mat[2, 2]
        self._get_pars()
        return

    
    def _set_by_mat(self, mat_input: np.ndarray) -> None:
        """
        ax^2 + bxy + cy^2 + dx + ey + f = 0 \n
        x.T * C * x = 0 \n
        C = \n
        [[  a, b/2, d/2], \n
         [b/2,   c, e/2], \n
         [d/2, e/2,   f]] \n
        """
        if mat_input.size == 9:
            if mat_input.shape != (3, 3):
                mat_input = mat_input.reshape(3, 3)
            if mat_input[-1, -1] != 0:
                mat_input = mat_input / mat_input[-1, -1]
            self.mat = mat_input
            self.__refresh()

        else:
            raise TypeError("Line2D init by not support type (3*3).")
        return

    def _set_by_5points2d(self, points2d: np.ndarray) -> None:
        """
        A * x = 0 \n
        A = \n
        [u^2, uv, v^2, u, v, 1] \n
        ...                     \n
        [  0,  0,   0, 0, 0, 1] \n
        
        x = \n
        [  a,  b,   c, d, e, f] 
        """
        # check
        if len(points2d.shape) == 1:
            raise IndexError("点太少")
        else:
            if points2d.shape[0] < 5:
                raise IndexError("点太少")

        n_points = points2d.shape[0]
        A = np.zeros((n_points + 1, 6))
        for [idx, point2d] in enumerate(points2d):
            [u, v] = point2d
            A[idx, :] = np.array([u ** 2, u * v, v ** 2, u, v, 1])
        A[-1, :] = np.array([0, 0, 0, 0, 0, 1])
        x = np.zeros(n_points + 1, )
        x[-1] = 1
        vec = np.linalg.lstsq(a=A, b=x, rcond=-1)[0]
        mat = self.__vec_to_mat(vec)
        self._set_by_mat(mat)
        return self

    def _set_by_vec(self, vec: np.ndarray) -> None:
        mat = self.vector_to_mat(vec)
        self._set_by_mat(mat)
        return 

    def _set_by_6pars(self, a: float, b: float, c: float, d: float, e: float, f: float) -> None:
        """
        ax^2 + bxy + cy^2 + dx + ey + f = 0
        """
        vec = np.array([a, b, c, d, e, f])
        mat = self.__vec_to_mat(vec)
        self._set_by_mat(mat)
        return 

    def __str__(self) -> str:
        str1 = self.mat.__str__() + "\n"
        str2 = "{}x^2 + {}xy + {}y^2 + {}x + {}y + {} = 0".format(self.a, self.b, self.c, self.d, self.e, self.f) + "\n"
        str3 = "center: ({:3f}, {:3f}), \tradius: ({:3f}, {:3f}), \ttheta_angle: {:3f}".format(self.point2d_center[0], self.point2d_center[1], self.radius_u, self.radius_v, 180 * self.theta_rad / np.pi)

        return  str1 + str2 + str3
            

    def __mat_to_vec(self, mat):            
        a = mat[0, 0]
        b = mat[0, 1] * 2
        c = mat[1, 1]
        d = mat[2, 0] * 2
        e = mat[2, 1] * 2
        f = mat[2, 2]
        vec = np.array([a, b, c, d, e, f])
        return vec
    
    def __vec_to_mat(self, vec):
        [a, b, c, d, e, f] = vec
        mat = np.diag([a, c, f])
        mat[0, 1] = b
        mat[0, 2] = d
        mat[1, 2] = e
        mat = (mat + mat.T) / 2
        return mat

    # def get_tangent_line(self, point_tangent):
    #     mat_line = get_tangent_line_of_conic(self.mat, point_tangent.mat)    
    #     return Line2D(mat_line)

    def transform_by_homography_mat(self, mat_homography) -> None:
        mat_transed = get_transformed_conic_mat(self.mat, mat_homography)
        self.mat = mat_transed
        self.__refresh()
        return 

    def draw(self, img: np.ndarray, color=(255, 0, 0), thickness=1) -> None:
        self._get_pars()
        pars_plot = (tuple(self.point2d_center), (self.radius_u * 2, self.radius_v * 2),  self.theta_drawn)
        if np.isnan(pars_plot[1][0]) or np.isnan(pars_plot[1][1]):
            return
        else:
            cv2.ellipse(img, pars_plot, color=color, thickness=thickness) #画椭圆
            return

    def _get_pars(self):
        """
        [x, y] @ R = [x_, y_]
        [d, e] @ R = [d_, e_]
        evec0 * x_^2 + evec1 * y_^2 + d_ * x_ + e_ * y_ + f = 0
        evec0 * ((x_ + s0)^2 - s0^2) + evec1 * ((y_ + s1)^2 - s1^2) + f = 0
        s0 = 0.5 * d_ / evec0; s1 = 0.5 * e_ / evec1 
        u = evec0 * s0^2 + evec1 * s1^2 - f 
        """
        evec, R_ = np.linalg.eig(self.mat[:2, :2]) # R.I = R.T , orthogonality
        [d_, e_] = np.array([self.d, self.e]) @ R_

        s0 = 0.5 * d_ / evec[0]
        s1 = 0.5 * e_ / evec[1]
        k = evec[0] * (s0 ** 2) + evec[1] * (s1 ** 2) - self.f 

        self.point2d_center = np.array([-s0, -s1]) @ R_.T
        self.radius_u = np.sqrt(np.abs(k / evec[0]))
        self.radius_v = np.sqrt(np.abs(k / evec[1]))

        arc_c = np.arccos(R_[0, 0])
        arc_s = np.arcsin(R_[0, 1])
        if   np.abs(arc_c - arc_s) < eps.eps:
            self.theta_rad =  arc_c
        elif np.abs(arc_c - arc_s) > eps.eps and np.abs(arc_c + arc_s) > eps.eps:
            arc_s = np.arcsin(R_[0, 0])
            arc_c = np.arccos(R_[0, 1])
            self.radius_u = np.sqrt(k / evec[1])
            self.radius_v = np.sqrt(k / evec[0])
            self.theta_rad =  arc_c
        elif np.abs(arc_c - arc_s) > eps.eps and np.abs(arc_c + arc_s) < eps.eps:
            arc_s = np.arcsin(-R_[0, 0])
            arc_c = np.arccos(-R_[0, 0])
            self.theta_rad =  arc_c
        self.theta_drawn = - self.theta_rad / np.pi * 180
        [dx, dy] = self.point2d_center
        dr = self.theta_rad
        [da ,db] = [self.radius_u, self.radius_v]

        R = np.array([
            [ np.cos(dr), np.sin(dr), 0], 
            [-np.sin(dr), np.cos(dr), 0], 
            [          0,          0, 1]
        ])
        T = np.array([
            [1, 0, dx], 
            [0, 1, dy], 
            [0, 0,  1]
        ])
        S = np.diag([-da, -db, 1])
        self.mat_R = R
        self.mat_T = T
        self.mat_S = S
        self.mat_H = T @ R @ S
        return 


class Conic2dStd(Conic2d):
    def __init__(self) -> None:
        input_mat = np.array([
            [-1,  0, 0],
            [ 0, -1, 0],
            [ 0,  0, 1]
        ])
        self._set_by_mat(input_mat)
        return


def get_tangent_line_of_conic(mat_conic, mat_point_tangent):
    mat_line = mat_conic @ mat_point_tangent
    return mat_line

def get_transformed_conic_mat(mat_conic, mat_homography):
    """
    x.T * C * x = 0 transform by homography ->
    x2.T * H.T.I * C * H.I * x2 = 0 ->
    x2.T * C2 * x2 = 0
    """
    H_inv = np.linalg.inv(mat_homography)
    mat_coic_tarns = H_inv.T @ mat_conic @ H_inv
    return mat_coic_tarns

def get_homography_mat_from_2conics(conic_source, conic_target):
    """
    H.I.T @ conic_source @ H.I = conic_target
    """
    H0 = conic_source.mat_T @ conic_source.mat_R @ conic_source.mat_S
    H1 = conic_target.mat_T @ conic_target.mat_R @ conic_target.mat_S
    H =  H1 @ np.linalg.inv(H0)
    return H


def m_iou(mask1, mask2):
    i = np.where(mask1 * mask2 != 0)[0].shape[0]
    u = np.where(mask1 + mask2 != 0)[0].shape[0]
    return (i / u)

def d_iou(mask1, mask2):
    
    idxs1 = np.flip(np.array(np.where(mask1 != 0)).T)
    idxs2 = np.flip(np.array(np.where(mask2 != 0)).T)
    center1 = np.array([np.average(idxs1[:, 0]), np.average(idxs1[:, 1])]).astype(np.int0)
    center2 = np.array([np.average(idxs2[:, 0]), np.average(idxs2[:, 1])]).astype(np.int0)

    rect1 = cv2.minAreaRect(idxs1)
    box1 = cv2.boxPoints(rect1).astype(np.int0)
    rect2 = cv2.minAreaRect(idxs2)
    box2 = cv2.boxPoints(rect2).astype(np.int0)
    rect3 = cv2.minAreaRect(np.vstack((idxs1, idxs2)))
    box3 = cv2.boxPoints(rect3).astype(np.int0)

    delta1 = center1 - center2
    d_square = np.dot(delta1, delta1)
    delta2 = box3[0] - box3[2]
    r_square = np.dot(delta2, delta2)
    
    d_iou = m_iou(mask1, mask2) #- d_square #/ r_square

    return d_iou

def ellipse_loss_func(rtvec, args: List):
    M = args[0]
    points3d = args[1]
    points2d_obj = args[2]
    points2d_pro = geo.project_points3d_to_2d(rtvec, M, points3d)
    C = Ellipse2d()
    E = Ellipse2d()
    C._set_by_5points2d(points2d_obj)
    E._set_by_5points2d(points2d_pro)
    
    delta = (C.point2d_center - E.point2d_center)
    
    mask1 = np.zeros((1280*2, 800*2))
    mask2 = np.zeros((1280*2, 800*2))
    C.draw(mask1, thickness=-1)
    E.draw(mask2, thickness=-1)

    # DEBUG
    # debug1 = np.zeros((640, 480, 3))
    # debug2 = np.zeros((640, 480, 3))
    # C.draw(debug1, thickness=2, color=(0, 255, 0))
    # E.draw(debug2, thickness=2, color=(255, 0, 0))
    # cv2.namedWindow("DEBUG", flags=cv2.WINDOW_FREERATIO)
    # cv2.imshow("DEBUG", debug1 + debug2)
    # cv2.waitKey(1)
    
    #l2 = np.linalg.norm(points2d_pro[0] - points2d_obj[0])
    #d = E.point2d_center - C.point2d_center
    #d_square = np.dot(d, d.T)
    loss = 1 - d_iou(mask1, mask2) #+ np.linalg.norm(d) #/ (d_square**0.5 + C.radius_u + E.radius_u)#+ l2 
    return loss


def ellipse_loss_func_multi(rtvec: np.ndarray, args: List):
    """
    rtvec, [mat_projection, points3d_project, points2d_object]
    points3d_project[0 ]: 顶点
    points3d_project[1:]: 椭圆点
    """
    Ms = args[0]
    points3d = args[1]
    points2d_obj = args[2]

    n_cams = len(Ms)
    loss_total = 0
    for i_cam in range(n_cams):
        loss = ellipse_loss_func(rtvec, [Ms[i_cam], points3d, points2d_obj[i_cam]])
        loss_total += loss
        
        # DEBUG
        debug1 = FileIO.imread("C:/Users/Li/Desktop/Pose6dSolver-pyqt/姿态测量/images_solve/cam_1/scene_2.png")
        debug2 = FileIO.imread("C:/Users/Li/Desktop/Pose6dSolver-pyqt/姿态测量/images_solve/cam_2/scene_2.png")
        Visualizer.draw_points2d([debug1, debug2][i_cam], points2d_obj[i_cam], radius=1)

        C = Ellipse2d()
        C._set_by_5points2d(points2d_obj[i_cam])
        C.draw([debug1, debug2][i_cam], thickness=2, color=(0, 255, 0))
        
        E = Ellipse2d()
        E._set_by_5points2d(geo.project_points3d_to_2d(rtvec, Ms[i_cam], points3d))
        E.draw([debug1, debug2][i_cam], thickness=2, color=(255, 0, 0))
        cv2.namedWindow("DEBUG{}".format(i_cam), flags=cv2.WINDOW_FREERATIO)
        cv2.imshow("DEBUG{}".format(i_cam), [debug1, debug2][i_cam])
        cv2.waitKey(1)
    return loss_total / n_cams

def get_jacobian_matrix(params, func_objective, args_of_func_objective):
    """
    params, func_objective, args_of_func_objective
    """
    delta = 1E-3
    n_objects = 1
    n_params = params.shape[0]
    J = np.zeros(n_params)
    for [idx_parm, param] in enumerate(params):
        params_delta_p = params.copy()
        params_delta_n = params.copy()
        params_delta_p[idx_parm] = param + delta
        params_delta_n[idx_parm] = param - delta

        loss_delta_p = func_objective(params_delta_p, args_of_func_objective)
        loss_delta_n = func_objective(params_delta_n, args_of_func_objective)
        dl_of_dp = (loss_delta_p - loss_delta_n) / (2 * delta)
        J[idx_parm] = dl_of_dp
    return J

