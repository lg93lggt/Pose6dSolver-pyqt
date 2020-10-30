
import enum
import sys
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
sys.path.append("..")
from core import eps 
from core import geometry as geo
from core import Conic


class Ellipse2d(Conic.Conic2d):
    """
    Conic(input_mat):           \\
        init by mat C           \\
        x.T * C * x = 0         \\
                                \\
    input argument:             \\
        input_mat:              \\
            type:   np.ndarray  \\
            shape:  3x3
    """
    def __init__(self, input_mat: np.ndarray=None) -> None:
        """
        input argument: 
            input_mat: 
                type:   np.ndarray 
                shape:  3x3
        """
        if input_mat is None:
            self.mat = np.zeros((3, 3))
        else:
            if type(input_mat) is np.ndarray:
                self._set_by_mat(input_mat=input_mat)
        return

    def __refresh(self):
        if self.mat[-1, -1] != 0:
            self.mat = -self.mat / self.mat[-1, -1]
        self.a = self.mat[0, 0]
        self.b = self.mat[0, 1] * 2
        self.c = self.mat[1, 1]
        self.d = self.mat[2, 0] * 2
        self.e = self.mat[2, 1] * 2
        self.f = self.mat[2, 2]
        self._get_pars()
        return

    def _set_by_mat(self, input_mat: np.ndarray) -> None:
        """
        ax^2 + bxy + cy^2 + dx + ey + f = 0 \\
        x.T * C * x = 0                     \\
                                            \\
        C =                                 \\
        [[  a, b/2, d/2],                   \\
         [b/2,   c, e/2],                   \\
         [d/2, e/2,   f]]                   
        """
        if input_mat.size == 9:
            if input_mat.shape != (3, 3):
                input_mat = input_mat.reshape(3, 3)
            self.mat = input_mat
            self.__refresh()

        else:
            raise TypeError("Line2D init by not support type (3*3).")
        return

    def _set_by_5points2d(self, points2d: np.ndarray) -> None:
        """
        A * x = 0               \\
                                \\
        A =                     \\
        [u^2, uv, v^2, u, v, 1] \\
        ...                     \\
        [  0,  0,   0, 0, 0, 1] \\
                                \\
        x =                     \\
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
        x = np.zeros(6, )
        x[-1] = 1
        vec = np.linalg.lstsq(a=A, b=x, rcond=-1)[0]
        mat = self.__vec_to_mat(vec)
        mat = eps.filter(mat)
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
        [x, y] @ R = [x_, y_]                                               \\
        [d, e] @ R = [d_, e_]                                               \\
        evec0 * x_^2 + evec1 * y_^2 + d_ * x_ + e_ * y_ + f = 0             \\
        evec0 * ((x_ + s0)^2 - s0^2) + evec1 * ((y_ + s1)^2 - s1^2) + f = 0 \\
        s0 = 0.5 * d_ / evec0; s1 = 0.5 * e_ / evec1                        \\
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

    def transform_by_homography_mat(self, mat_homography) -> None:
        mat_transed = get_transformed_conic_mat(self.mat, mat_homography)
        self.mat = mat_transed
        self.__refresh()
        return 

    def get_transformed_conic(self, mat_homography) -> None:
        mat_transed = get_transformed_conic_mat(self.mat, mat_homography)
        new_conic = Ellipse2d(mat_transed)
        return new_conic

    def cal_angle_from_point2d_to_center_on_ellipse(self, point2d: List or np.ndarray) -> float:
        TI = np.linalg.inv(self.mat_T)
        # check
        
        A = self.radius_u + eps.eps
        # B = self.radius_v + eps.eps
        [u_, v_, _] = TI @ np.array([point2d[0], point2d[1], 1]).reshape(-1, 1)
        if   v_ > 0:
            angle = np.arccos(u_ / A)
        elif v_ < 0:
            angle = 2*np.pi - np.arccos(u_ / A)
        else: # v_ = 0
            angle = 0
        return angle

    def cal_sector_area(self, point2d_src: List or np.ndarray, point2d_dst: List or np.ndarray):
        A = self.radius_u + eps.eps
        B = self.radius_v + eps.eps
        theta1 = self.cal_angle_from_point2d_to_center(point2d_src)
        theta2 = self.cal_angle_from_point2d_to_center(point2d_dst)

        if   theta1 < theta2:
            area_sector = (theta2 - theta1) * A * B / 2
        elif theta1 > theta2:
            area_sector = np.pi * A * B - (theta1 - theta2) * A * B / 2
        else: # v_ = 0
            area_sector =  0
        return area_sector

    def cal_segment_area(self, point2d_src: List or np.ndarray, point2d_dst: List or np.ndarray):
        theta1 = self.cal_angle_from_point2d_to_center(point2d_src)
        theta2 = self.cal_angle_from_point2d_to_center(point2d_dst)
        area_sector = self.cal_sector_area(point2d_src, point2d_dst)
        # B = self.radius_v + eps.eps
        [u1, v1] = point2d_src
        [u2, v2] = point2d_dst

        if   theta1 < theta2:
            area_segment = area_sector - 0.5*np.abs(u1*v2 - u2*v1)
        elif theta1 > theta2:
            area_segment = area_sector + 0.5*np.abs(u1*v2 - u2*v1)
        else: # theta1 = theta2:
            area_segment = 0
        return area_segment


class Ellipse2dStd(Ellipse2d):
    def __init__(self, input_vec: np.ndarray=None) -> None:
        if input_vec is None:
            mat = np.array([
                [1,  0,  0],
                [0,  1,  0],
                [0,  0, -1]
            ])
        else:
            if input_vec.shape == (2, ):
                mat = np.diag(np.append(input_vec, -1))
            else:
                raise IndexError("shape should be (3, )")
        self._set_by_mat(mat)
        return

    def cal_sector_area(self, point2d_src, point2d_dst):
        A = self.radius_u + eps.eps
        B = self.radius_v + eps.eps
        [u1, v1] = point2d_src
        [u2, v2] = point2d_dst
        if v1 >= 0:
            theta1 = np.arccos(u1 / A)
        else:
            theta1 = 2*np.pi - np.arccos(u1 / A)

        if v2 >= 0:
            theta2 = np.arccos(u2 / A)
        else:
            theta2 = 2*np.pi - np.arccos(u2 / A)

        if   theta1 < theta2:
            area_sector = (theta2 - theta1) * A * B / 2
        elif theta1 > theta2:
            area_sector = np.pi * A * B - (theta1 - theta2) * A * B / 2
        else:
            area_sector =  0
        return area_sector

    def cal_segment_area(self, point2d_src, point2d_dst):
        area_sector = self.cal_sector_area(point2d_src, point2d_dst)
        A = self.radius_u + eps.eps
        B = self.radius_v + eps.eps
        [u1, v1] = point2d_src
        [u2, v2] = point2d_dst
        if v1 >= 0:
            theta1 = np.arccos(u1 / A)
        else:
            theta1 = 2*np.pi - np.arccos(u1 * self.a)

        if v2 >= 0:
            theta2 = np.arccos(u2 / A)
        else:
            theta2 = 2*np.pi - np.arccos(u2 / A)

        if   theta1 < theta2:
            area_segment = area_sector - 0.5*np.abs(u1*v2 - u2*v1)
        elif theta1 > theta2:
            area_segment = area_sector + 0.5*np.abs(u1*v2 - u2*v1)
        else:
            area_segment = 0
        return area_segment


def cal_characteristic_polynomial_of_pencil_between_2ellipses(ellipse_src, ellipse_dst):
    [a1, b1, c1, d1, e1, f1] = [ellipse_src.a, ellipse_src.b, ellipse_src.c, ellipse_src.d, ellipse_src.e, ellipse_src.f]
    [a2, b2, c2, d2, e2, f2] = [ellipse_dst.a, ellipse_dst.b, ellipse_dst.c, ellipse_dst.d, ellipse_dst.e, ellipse_dst.f]
    
    _d = a1*(c1*f1 - e1**2) - (c1 * d1**2 - 2*b1*d1*e1 + b1**2 * f1)
    _a = ( \
        a1*(c1*f2 - 2*e1*e2 + c2*f1) + \
        2*b1*(d1*e2 - b2*f1 + d2*e1) + \
        2*d1*(b2*e1 - c1*d2) - \
        (b1**2 * f2 + c2 * d1**2 + a2 * e1**2) + \
        (a2*c1*f1) \
        ) / _d
    _b = ( \
        a1*(c2*f2 - e1**2) + \
        2*b1*(d2*e2 - b2*f2) + \
        2*d1*(b2*e2 - c2*d2) + \
        c1*(a2*f2 - d2**2) + \
        2*e1*(b2*d2 - a2*e2) + \
        f1*(a2*c2 - b2**2) \
    ) / _d
    _c = (
        a2*(c2*f2 - e2**2) - \
        (b2**2 * f2 - 2*b2*d2*e2 + c2 * d2**2)
    )
    return [_a, _b, _c]

class RelativePositionOfTwoEllipses(enum.Enum):
    COINCIDENT                               = 0
    TRANSVERSAL_AT_4_PTS                     = 1
    TRANSVERSAL_AT_2_PTS                     = 2
    SEPARATED                                = 3
    CONTAINED                                = 4
    TRANSVERSAL_AT_2_PTS_AND_TANGENT_AT_1_PT = 5
    EXTERNALLY_TANGENT_AT_1_PT               = 6
    INTERNALLY_TANGENT_AT_1_PT               = 7
    INTERNALLY_TANGENT_AT_2_PTS              = 8
    TRANSVERSAL_AT_1_PT_AND_TANGENT_AT_1_PT  = 9

def classify_relative_position_between_2ellipes(ellipse_src, ellipse_dst):
    [_a, _b, _c] = cal_characteristic_polynomial_of_pencil_between_2ellipses(ellipse_src, ellipse_dst)
    s4 = -27 * _c**3 + 18*_a*_b*_c + _a**2 * _b**2 - 4 * _a**3*_c - 4 * _b**3
    s1 = _a
    s2 = _a**2 - 3*_b
    s3 = 3*_a*_c + _a**2 * _b - 4 * _b**2
    if   s4 < 0: # case 2
        return RelativePositionOfTwoEllipses.TRANSVERSAL_AT_2_PTS
    elif s4 > 0:
        if (s1 > 0) and (s2 > 0) and (s3 > 0): # case 1, 4
            _u =( -_a - np.sqrt(s2)) / 3
            _v =( -_a + np.sqrt(s2)) / 3
            _M = _u * ellipse_src.mat + ellipse_dst.mat
            _N = _v * ellipse_src.mat + ellipse_dst.mat
            if ( \
                ((_M[1, 1] * np.linalg.det(_M) > 0) and (_M[0, 0] * np.linalg.det(_M[1:, 1:]) > 0)) or \
                ((_N[1, 1] * np.linalg.det(_M) > 0) and (_N[0, 0] * np.linalg.det(_N[1:, 1:]) > 0))
            ): # case 4
                return RelativePositionOfTwoEllipses.CONTAINED
            else: # case 1
                return RelativePositionOfTwoEllipses.TRANSVERSAL_AT_4_PTS
        else: # case 3
            return RelativePositionOfTwoEllipses.SEPARATED
    else:
        if (s1 > 0) and (s2 > 0) and (s3 < 0): # case 6
            return RelativePositionOfTwoEllipses.EXTERNALLY_TANGENT_AT_1_PT
        elif (s1 > 0) and (s2 > 0) and (s3 > 0): # case 4, 5, 7, 8
            alpha = (4*_a*_b - _a**3 - 9*_c) / (s2) 
            beta  = (9*_c - _a*_b) / (s2) 
            _M = beta  * ellipse_src.mat + ellipse_dst.mat
            _N = alpha * ellipse_src.mat + ellipse_dst.mat
            if np.linalg.det(_M[:-1, :-1]) > 0:
                if np.linalg.det(_N[:-1, :-1]) > 0: # case 5
                    return RelativePositionOfTwoEllipses.TRANSVERSAL_AT_2_PTS_AND_TANGENT_AT_1_PT
                else: # case 7
                    return RelativePositionOfTwoEllipses.INTERNALLY_TANGENT_AT_1_PT
            elif (np.linalg.det(_M[1:, 1:]) + np.linalg.det(_M[[0, 2], [0, 2]]))> 0: # case 4
                return RelativePositionOfTwoEllipses.CONTAINED
            else: # case 8
                return RelativePositionOfTwoEllipses.INTERNALLY_TANGENT_AT_2_PTS
        else: # case 7, 9, 0
            alpha = -_a / 3
            _M = alpha * ellipse_src.mat + ellipse_dst.mat
            if ( \
                ((np.linalg.det(_M) != 0) and (np.linalg.det(_M[:-1, :-1]) <= 0)) or \
                ( \
                    (np.linalg.det(_M) != 0) and \
                    (np.linalg.det(_M[:-1, :-1]) <= 0) and \
                    ((np.linalg.det(_M) / (_M[0, 0] + _M[1, 1])) < 0)
                )
            ): # case 0
                return RelativePositionOfTwoEllipses.COINCIDENT
            elif ((np.linalg.det(_M) != 0) and (np.linalg.det(_M[:-1, :-1]) > 0)): # case 9
                return RelativePositionOfTwoEllipses.TRANSVERSAL_AT_1_PT_AND_TANGENT_AT_1_PT
            else: # case 7
                return RelativePositionOfTwoEllipses.INTERNALLY_TANGENT_AT_1_PT

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
    mat_conic_tarns = H_inv.T @ mat_conic @ H_inv
    mat_conic_tarns = -mat_conic_tarns / mat_conic_tarns[-1, -1]
    return mat_conic_tarns

def get_homography_mat_from_2conics(conic_source, conic_target):
    """
    H.I.T @ conic_source @ H.I = conic_target
    """
    H0 = conic_source.mat_T @ conic_source.mat_R @ conic_source.mat_S
    H1 = conic_target.mat_T @ conic_target.mat_R @ conic_target.mat_S
    H =  H1 @ np.linalg.inv(H0)
    return H



def main(args):
    def ellipse_loss_func(rtvec: np.ndarray, args: List):
        """
        rtvec, [mat_projection, points3d_project, conic2d_object]
        """
        M = args[0]
        points3d = args[1]
        C = args[2]
        points2d = geo.project_points3d_to_2d(rtvec, M, points3d)
        E = Ellipse2d()
        E._set_by_5points2d(points2d)
        
        delta = (C.point2d_center - E.point2d_center)
        
        img1 = np.zeros((480, 640))
        img2 = np.zeros((480, 640))
        C.draw(img1,thickness=-1)
        E.draw(img2,thickness=-1)
        from new_method import d_iou
        diou = d_iou(img1, img2)
        loss = 1-diou
        return loss

    def ellipse_loss_func2(rtvec: np.ndarray, args: List):        
        """
        rtvec, [mat_projection, points3d_project, points2d_object]
        """
        Ms = args[0]
        points3d = args[1]
        points2d_obj_for_all = args[2]
        def loss_fuc(rtvec, M, points2d_obj):
            points2d_pro = geo.project_points3d_to_2d(rtvec, M, points3d)
            C = Ellipse2d()
            E = Ellipse2d()
            C._set_by_5points2d(points2d_obj[1:])
            E._set_by_5points2d(points2d_pro[1:])
            D = (np.triu(C.mat) - np.triu(E.mat)) / C.mat
            l2 = np.linalg.norm(points2d_pro[0] - points2d_obj[0]) 
            return np.linalg.norm(D) + l2
        n_cams = 2
        loss_total = 0
        for i_cam in range(n_cams):
            loss = loss_fuc(rtvec, Ms[i_cam], points2d_obj_for_all[i_cam])
            loss_total += loss
            
        return loss_total / n_cams

    def ellipse_loss_func3(rtvec: np.ndarray, args: List):
        """
        rtvec, [mat_projection, points3d_project, points2d_object]
        """
        Ms = args[0]
        points3d = args[1]
        points2d_obj_for_all = args[2]
        def loss_fuc(rtvec, M, points2d_obj):
            points2d_pro = geo.project_points3d_to_2d(rtvec, M, points3d)
            C = Ellipse2d()
            E = Ellipse2d()
            C._set_by_5points2d(points2d_obj[1:])
            E._set_by_5points2d(points2d_pro[1:])
            
            delta = (C.point2d_center - E.point2d_center)
            
            mask1 = np.zeros((480, 640))
            mask2 = np.zeros((480, 640))
            C.draw(mask1, thickness=-1)
            E.draw(mask2, thickness=-1)
            from new_method import d_iou, m_iou
            
            l2 = np.linalg.norm(points2d_pro[0] - points2d_obj[0])
            loss = 1 - d_iou(mask1, mask2) + l2 
            return loss

        n_cams = 2
        loss_total = 0
        for i_cam in range(n_cams):
            loss = loss_fuc(rtvec, Ms[i_cam], points2d_obj_for_all[i_cam])
            loss_total += loss
            
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

if __name__ == "__main__":
    from core import FileIO
    from core import Visualizer 
    
    vis = Visualizer.Visualizer()
    fio = FileIO.FileIO()
    fio.load_project_from_filedir("../../姿态测量")

    cam = fio.load_camera_pars(0)
    mat_proj = cam["intrin"] @ cam["extrin"]

    p2d_obj = fio.load_points2d("solve", 0, 0, 0)

    p3d_src = fio.loadz_points3d("solve", 0, 0, 0)["array"]
    p2d_src = geo.project_points3d_to_2d(np.array([0., 0, 0, 0.01, -0.02, 0]), mat_proj, p3d_src)

    img = fio.load_image_raw("solve", 0, 0)


    ellipse_obj = Ellipse2d()
    ellipse_obj._set_by_5points2d(p2d_obj[1:])

    ellipse_src = Ellipse2d()
    ellipse_src._set_by_5points2d(p2d_src[1:])

    ellipse_obj = Ellipse2d()
    ellipse_obj._set_by_5points2d(p2d_obj[1:])

    T_obj = np.eye(3)
    T_obj[:2, -1] = -ellipse_obj.point2d_center
    R_obj = geo.rotation2d(-ellipse_obj.theta_rad)
    ellipse_obj_rt = ellipse_obj.get_transformed_conic(R_obj @ T_obj)
    ellipse_src_rt = ellipse_src.get_transformed_conic(R_obj @ T_obj)

    ellipse_std = Ellipse2dStd(np.array([1/(4**2), 1/(2**2)]))

    ellipse_obj_rt.transform_by_homography_mat(np.array([[1,0,320.],[0,1,240],[0,0,1]]))
    ellipse_src_rt.transform_by_homography_mat(np.array([[1,0,320.],[0,1,240],[0,0,1]]))
    # ellipse_std.transform_by_homography_mat(np.array([[1,0,320.],[0,1,240],[0,0,1]]))

    ellipse_src.draw(img, color=(0, 0, 255), thickness=1)
    ellipse_obj.draw(img, color=(0, 255, 0), thickness=1)
    ellipse_obj_rt.draw(img, color=(0, 255, 255), thickness=1)
    ellipse_src_rt.draw(img, color=(255, 0, 255), thickness=1)
    ellipse_std.draw(img, color=(255, 255, 255), thickness=1)

    area1 = ellipse_std.cal_segment_area(
        [4/np.sqrt(5), 4/np.sqrt(5)], [-3, -np.sqrt(7)/2]
    )

    area2 = ellipse_std.cal_segment_area(
        [-3, -np.sqrt(7)/2], [4/np.sqrt(5), 4/np.sqrt(5)]
    )
    cal_characteristic_polynomial_of_pencil_between_2ellipses(ellipse_src_rt, ellipse_obj_rt)0000000000000
    cv2.namedWindow("", cv2.WINDOW_FREERATIO)
    cv2.imshow("", img)
    cv2.waitKey()
    print()
            

