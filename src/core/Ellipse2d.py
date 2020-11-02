
import enum
import sys
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import sympy as sp
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
            self.mat = np.diag([1, 1, -1.0])
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
        x = np.zeros(n_points + 1, )
        x[-1] = -1
        vec = np.linalg.lstsq(a=A, b=x, rcond=-1)[0]
        mat = self.__vec_to_mat(vec)
        mat = eps.filter(mat)
        self._set_by_mat(mat)
        return self

    def _set_by_5props(self, radius_u: float=1, radius_v: float=1, u_center: float=0, v_center: float=0, theta: float=0) -> None:
        S = np.diag([radius_u, radius_v, 1])
        T = np.array([
                [1, 0, u_center],
                [0, 1, v_center],
                [0, 0,        1]
            ])
        R = np.array([
                [ np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [             0,             0, 1]
            ])
        H = T @ R @ S
        mat = get_transformed_conic_mat(Ellipse2dStd().mat, H)
        self._set_by_mat(mat)
        return 

    def _set_by_6pars(self, a: float=1, b: float=0, c: float=1, d: float=0, e: float=0, f: float=-1) -> None:
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

        s0 = 0.5 * d_ / (evec[0] + eps.eps**2)
        s1 = 0.5 * e_ / (evec[1] + eps.eps**2)
        k = evec[0] * (s0 ** 2) + evec[1] * (s1 ** 2) - self.f 

        self.point2d_center = np.array([-s0, -s1]) @ R_.T
        self.radius_u = np.sqrt(np.abs(k / (evec[0] + eps.eps**2)))
        self.radius_v = np.sqrt(np.abs(k / (evec[1] + eps.eps**2)))

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
    l = sp.symbols("l")
    A = l * ellipse_src.mat
    B = ellipse_dst.mat
    polynomial = sp.det(sp.Matrix(A + B)).expand()
    if polynomial.coeff(l**3) == 0:
        raise ZeroDivisionError
    polynomial /= polynomial.coeff(l**3)
    [_, a, b, c] = sp.Poly(polynomial, l).coeffs()
    return [1, float(a), float(b), float(c)]

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
    [_, a, b, c] = cal_characteristic_polynomial_of_pencil_between_2ellipses(ellipse_src, ellipse_dst)
    
    # solve: x^3 + _a*x^2 + _b*x^1 + _c = 0
    roots = np.roots([1, a, b, c])
    
    # l = sp.symbols("l")
    # poly = l**3 + a*l**2 + b*l + c
    # roots = sp.solve()

    real_roots = get_real_roots(roots)
    n_real_roots = (np.abs(roots.imag) < eps.eps).sum()
    n_imag_roots = 3 - n_real_roots

    positive_real_roots = real_roots[np.where(real_roots > 0)]
    negative_real_roots = real_roots[np.where(real_roots < 0)]

    n_positive_real_roots = (real_roots > 0).sum()
    n_negative_real_roots = n_real_roots - n_positive_real_roots 
    
    print("p_r:{}\tn_r:{}".format(positive_real_roots, negative_real_roots))
    
    if (n_positive_real_roots == 2) and (n_negative_real_roots == 1):
        # When r1 < 0 < r2 < r3 ∈ R
        # Two separated ellipses 
        if np.abs(positive_real_roots[0] - positive_real_roots[1]) > eps.eps:
            return RelativePositionOfTwoEllipses.SEPARATED # case 3
        # When r1 < 0 < r2 = r3 ∈ R
        # Two ellipses touching each other externally 
        else: 
            return RelativePositionOfTwoEllipses.EXTERNALLY_TANGENT_AT_1_PT # case 6
    
    # When r1 < 0 ∈ R, r2 ∈ C, r3 ∈ C
    # Two ellipses transversal in 2 points 
    elif (n_negative_real_roots == 1) and (n_imag_roots == 2):
        return RelativePositionOfTwoEllipses.TRANSVERSAL_AT_2_PTS # case 2

    elif n_negative_real_roots == 3:
        # When r1 = r2 = r3 < 0 ∈ R
        if (np.abs(negative_real_roots[0] - negative_real_roots[1]) < eps.eps) and \
                (np.abs(negative_real_roots[1] - negative_real_roots[2]) < eps.eps) and \
                (np.abs(negative_real_roots[2] - negative_real_roots[0]) < eps.eps):
            alpha = -a/3
            pencil = alpha*ellipse_src.mat + ellipse_dst.mat
            rank_pencil = np.rank(pencil)
            if rank_pencil == 3:
                return RelativePositionOfTwoEllipses.COINCIDENT # case 10
            if rank_pencil == 2:
                return RelativePositionOfTwoEllipses.TRANSVERSAL_AT_1_PT_AND_TANGENT_AT_1_PT # case 9
            if rank_pencil == 1:
                return RelativePositionOfTwoEllipses.INTERNALLY_TANGENT_AT_1_PT # case 7
        # When r1 < r2 < r3 < 0 ∈ R
        elif (np.abs(negative_real_roots[0] - negative_real_roots[1]) > eps.eps) and \
                (np.abs(negative_real_roots[1] - negative_real_roots[2]) > eps.eps) and \
                (np.abs(negative_real_roots[2] - negative_real_roots[0]) > eps.eps):
            # oringnal      polynomial: l^3 + a*l^2 +   b*l + c = 0
            # derivative of polynomial:       3*l^2 + 2*a*l + b = 0
            roots_deriv_polynomial = np.roots([3, 2*a, b])
            u = np.min(roots_deriv_polynomial)
            v = np.max(roots_deriv_polynomial)
            M = u*ellipse_src.mat + ellipse_dst.mat
            N = v*ellipse_src.mat + ellipse_dst.mat
            
            if ((M[1,1]*np.linalg.det(M) > 0) and (np.linalg.det(M[1:, 1:]))) or \
                    ((N[1,1]*np.linalg.det(N) > 0) and (np.linalg.det(N[1:, 1:]))):
                return RelativePositionOfTwoEllipses.CONTAINED # case 4
            else:
                return RelativePositionOfTwoEllipses.TRANSVERSAL_AT_4_PTS # case 1
        # When r1 < 0 ∈ R, r2 = r3 < 0 ∈ R, r1 != r2
        if (\
                ((np.abs(negative_real_roots[0] - negative_real_roots[1]) > eps.eps) and \
                ( \
                    (np.abs(negative_real_roots[0] - negative_real_roots[2]) < eps.eps) or \
                    (np.abs(negative_real_roots[1] - negative_real_roots[2]) < eps.eps)
                )) or \
                ((np.abs(negative_real_roots[1] - negative_real_roots[2]) > eps.eps) and \
                ( \
                    (np.abs(negative_real_roots[1] - negative_real_roots[0]) < eps.eps) or \
                    (np.abs(negative_real_roots[2] - negative_real_roots[0]) < eps.eps)
                )) or \
                ((np.abs(negative_real_roots[2] - negative_real_roots[0]) > eps.eps) and \
                ( \
                    (np.abs(negative_real_roots[0] - negative_real_roots[1]) < eps.eps) or \
                    (np.abs(negative_real_roots[2] - negative_real_roots[1]) < eps.eps)
                ))
            ):
            alpha = (4*a*b - a**3 - 9*c) /   (a**2 - 3*b)
            beta  = (9*c - a*b)          / 2*(a**2 - 3*b)
            pencil_alpha = alpha * ellipse_src.mat + ellipse_dst.mat
            pencil_beta  = beta  * ellipse_src.mat + ellipse_dst.mat
            rank_pencil_alpha = np.linalg.matrix_rank(pencil_alpha)
            rank_pencil_beta  = np.linalg.matrix_rank(pencil_beta)
            if 2 == 3:
                return RelativePositionOfTwoEllipses.CONTAINED # case 4
            if (rank_pencil_alpha == 2) and (rank_pencil_beta == 2):
                return RelativePositionOfTwoEllipses.TRANSVERSAL_AT_1_PT_AND_TANGENT_AT_1_PT # case 5
            if 2 == 1:
                return RelativePositionOfTwoEllipses.INTERNALLY_TANGENT_AT_1_PT # case 7
            if 2 == 1:
                return RelativePositionOfTwoEllipses.INTERNALLY_TANGENT_AT_1_PT # case 8
            
    elif n_real_roots == 1:
        pass
    # s4 = -27 * _c**3 + \
    #     18*_a*_b*_c + \
    #     _a**2*_b**2 - \
    #     4*(_a**3)*_c - \
    #     4*(_b**3)

    # s1 = _a
    # s2 = _a**2 - 3*_b
    # s3 = 3*_a*_c + _a**2 * _b - 4 * _b**2
    # if   s4 < 0: # case 2
    #     return RelativePositionOfTwoEllipses.TRANSVERSAL_AT_2_PTS
    # elif s4 > 0:
    #     if (s1 > 0) and (s2 > 0) and (s3 > 0): # case 1, 4
    #         _u =( -_a - np.sqrt(s2)) / 3
    #         _v =( -_a + np.sqrt(s2)) / 3
    #         _M = _u * ellipse_src.mat + ellipse_dst.mat
    #         _N = _v * ellipse_src.mat + ellipse_dst.mat
    #         if ( \
    #             ((_M[1, 1] * np.linalg.det(_M) > 0) and (_M[0, 0] * np.linalg.det(_M[1:, 1:]) > 0)) or \
    #             ((_N[1, 1] * np.linalg.det(_M) > 0) and (_N[0, 0] * np.linalg.det(_N[1:, 1:]) > 0))
    #         ): # case 4
    #             return RelativePositionOfTwoEllipses.CONTAINED
    #         else: # case 1
    #             return RelativePositionOfTwoEllipses.TRANSVERSAL_AT_4_PTS
    #     else: # case 3
    #         return RelativePositionOfTwoEllipses.SEPARATED
    # else:
        # if (s1 > 0) and (s2 > 0) and (s3 < 0): # case 6
        #     return RelativePositionOfTwoEllipses.EXTERNALLY_TANGENT_AT_1_PT
        # elif (s1 > 0) and (s2 > 0) and (s3 > 0): # case 4, 5, 7, 8
        #     alpha = (4*a*b - a**3 - 9*c) / (s2) 
        #     beta  = (9*c - a*b) / (s2) 
        #     _M = beta  * ellipse_src.mat + ellipse_dst.mat
        #     _N = alpha * ellipse_src.mat + ellipse_dst.mat
        #     if np.linalg.det(_M[:-1, :-1]) > 0:
        #         if np.linalg.det(_N[:-1, :-1]) > 0: # case 5
        #             return RelativePositionOfTwoEllipses.TRANSVERSAL_AT_2_PTS_AND_TANGENT_AT_1_PT
        #         else: # case 7
        #             return RelativePositionOfTwoEllipses.INTERNALLY_TANGENT_AT_1_PT
        #     elif (np.linalg.det(_M[1:, 1:]) + np.linalg.det(_M[[0, 2], [0, 2]]))> 0: # case 4
        #         return RelativePositionOfTwoEllipses.CONTAINED
        #     else: # case 8
        # #         return RelativePositionOfTwoEllipses.INTERNALLY_TANGENT_AT_2_PTS
        # # else: # case 7, 9, 0
        #     alpha = -a / 3
        #     _M = alpha * ellipse_src.mat + ellipse_dst.mat
        #     if ( \
        #                 ((np.linalg.det(_M) != 0) and (np.linalg.det(_M[:-1, :-1]) <= 0)) or \
        #                 ( \
        #                         (np.linalg.det(_M) != 0) and \
        #                         (np.linalg.det(_M[:-1, :-1]) <= 0) and \
        #                         ((np.linalg.det(_M) / (_M[0, 0] + _M[1, 1])) < 0)
        #                     )
        #             ): # case 0
        #         return RelativePositionOfTwoEllipses.COINCIDENT
        #     elif ((np.linalg.det(_M) != 0) and (np.linalg.det(_M[:-1, :-1]) > 0)): # case 9
        #         return RelativePositionOfTwoEllipses.TRANSVERSAL_AT_1_PT_AND_TANGENT_AT_1_PT
        #     else: # case 7
        #         return RelativePositionOfTwoEllipses.INTERNALLY_TANGENT_AT_1_PT

def get_tangent_line_of_conic(mat_conic, mat_point_tangent):
    mat_line = mat_conic @ mat_point_tangent
    return mat_line

def get_transformed_conic_mat(mat_conic, mat_homography):
    """
    x.T * C * x = 0 transform by homography ->
    (H * x2).T * C * (H * x2)
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

def get_real_roots(roots: np.ndarray or List[np.complex64 or np.complex128]):
    real_roots = roots[np.where(np.abs(roots.imag) < eps.eps)]
    # imag_roots = roots[np.where(np.abs(roots.real) < eps.eps)]
    return real_roots.real


if __name__ == "__main__":
    from core import FileIO
    from core import Visualizer 
    
    vis = Visualizer.Visualizer()
    fio = FileIO.FileIO()
    fio.load_project_from_filedir("../../姿态测量")

    img = np.zeros((800, 1280, 3), np.uint8)
    edges = cv2.Canny(img, 50, 500)
    img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # cnt = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # for i in range(len(cnt[1])):
    #     cv2.drawContours(img, cnt[1], i, (0,0,255))

    cam = fio.load_camera_pars(0)
    mat_proj = np.array([
        [-6.65331302e-01, -3.96516135e-02,  1.60672702e-01, -2.21357715e-01],
        [ 5.77521005e-02, -6.73878897e-01,  1.25742212e-01, -8.76790878e-02],
        [-1.18982088e-04, -9.53190398e-05, -2.70041496e-04, -3.97555260e-04],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

    p2d_obj = np.array([
        [282, 264],
        [690, 659],
        [613, 757],
        [624, 675],
        [675, 742],
        [703, 673],
        [657, 657]])

    p3d_src = np.array([
        [ 0.     ,  0.     ,  0.     ],
        [ 0.7    ,  0.02828,  0.03182],
        [ 0.7    , -0.02828, -0.03182],
        [ 0.7    ,  0.02828, -0.03182],
        [ 0.7    , -0.02828,  0.03182],
        [ 0.7    ,  0.04257,  0.     ]])
    p2d_src = geo.project_points3d_to_2d(np.array([0., 0, 0, -0.7, 0.2, -0.2]), mat_proj, p3d_src)
    vis.draw_axis3d(img, cam)
    vis.draw_points2d(img, p2d_obj)

    ellipse_obj = Ellipse2d()
    ellipse_obj._set_by_5points2d(p2d_obj[1:])

    ellipse_src = Ellipse2d()
    ellipse_src._set_by_5points2d(p2d_src[1:])

    ellipse_obj = Ellipse2d()
    ellipse_obj._set_by_5points2d(p2d_obj[1:])

    T_obj = np.eye(3)
    T_obj[:2, -1] = -ellipse_obj.point2d_center
    R_obj = geo.rotation2d(-ellipse_obj.theta_rad)
    S_obj = np.diag([1/ellipse_obj.radius_u, 1/ellipse_obj.radius_v, 1])
    ellipse_obj_rt = ellipse_obj.get_transformed_conic(R_obj @ T_obj)
    ellipse_src_rt = ellipse_src.get_transformed_conic(R_obj @ T_obj)


    ellipse_obj_rt.transform_by_homography_mat(np.array([[1, 0, img.shape[1] // 2],[0, 1, img.shape[0] // 2],[0, 0, 1]]))
    ellipse_src_rt.transform_by_homography_mat(np.array([[1, 0, img.shape[1] // 2],[0, 1, img.shape[0] // 2],[0, 0, 1]]))
    # ellipse_std.transform_by_homography_mat(np.array([[1,0,320.],[0,1,240],[0,0,1]]))

    ellipse_src.draw(img, color=(0, 0, 255), thickness=1)
    ellipse_obj.draw(img, color=(0, 255, 0), thickness=1)
    ellipse_obj_rt.draw(img, color=(0, 255, 255), thickness=1)
    ellipse_src_rt.draw(img, color=(255, 0, 255), thickness=1)
    

    #ellipse_src_rt.radius_u, ellipse_src_rt.radius_v, ellipse_src_rt.point2d_center[0], ellipse_src_rt.point2d_center[1], ellipse_src_rt.theta_rad)
    #ellipse_std.draw(img, color=(255, 255, 255), thickness=1)

    
    relation = classify_relative_position_between_2ellipes(ellipse_obj_rt, ellipse_src_rt)

    #e1  = Ellipse2d()._set_by_6pars(3, 2, 0, 0)
    e2  = Ellipse2d()
    print(relation.name, relation.value)
   
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()
    #cv2.namedWindow("", cv2.WINDOW_FREERATIO)
    # cv2.imshow("", img)
    # cv2.waitKey()
    print()
            

