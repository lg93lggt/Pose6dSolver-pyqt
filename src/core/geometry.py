
import json
import math

import cv2
from easydict import EasyDict
import numpy as np



def R_to_r(R: np.ndarray)-> np.ndarray:
    """
        旋转矩阵转向量
    """
    R_ = R[:3, :3]
    rvec = cv2.Rodrigues(R_)[0].flatten()
    return rvec

def r_to_R(rvec: np.ndarray)-> np.ndarray:
    """
        旋转向量转矩阵
    """
    R = np.eye(4)
    R_3x3 = cv2.Rodrigues(rvec)[0]
    R[:3,  :3] = R_3x3
    return R

def T_to_t(T: np.ndarray)-> np.ndarray:
    """
        平移矩阵转向量
    """
    tvec = T[:3, 3]
    return tvec

def t_to_T(tvec: np.ndarray)-> np.ndarray:
    """
        平移向量转矩阵
    """
    if tvec.size == 3:
        tvec = tvec.flatten()
    T = np.eye(4)
    T[:3, 3] = tvec
    return T

def rtmat_to_rtvec(RT: np.ndarray) -> np.ndarray:
    """
        位姿矩阵转向量
    """
    rtvec = np.zeros(6)
    R = np.eye(4)
    R[:3, :3] = RT[:3, :3]
    T = np.eye(4)
    T[:3,  3] = RT[:3,  3]
    rtvec[:3] = R_to_r(R)
    rtvec[3:] = T_to_t(T)
    return rtvec

def rtvec_to_rtmat(rtvec: np.ndarray) -> np.ndarray:
    """
        位姿向量转矩阵
    """
    rtvec = rtvec.reshape(6)
    R = r_to_R(rtvec[:3])
    T = t_to_T(rtvec[3:])
    return T @ R


def rtvec_degree2rad(rtvec_degree: np.ndarray) -> np.ndarray:
    """
        rtvec角度转弧度
    """
    rtvec_rad = rtvec_degree.copy()
    rtvec_rad[:3] = np.pi * (rtvec_rad[:3] / 180)
    return rtvec_rad
    
def rtvec_rad2degree(rtvec_rad: np.ndarray) -> np.ndarray:
    """
        rtvec弧度转角度
    """
    rtvec_degree = rtvec_rad.copy()
    rtvec_degree[:3] = 180 * (rtvec_degree[:3] / np.pi)
    return rtvec_degree

def solve_projection_mat_3d_to_2d(points3d: np.ndarray, points2d: np.ndarray, method="svd")-> np.ndarray:
    """
        解3d-2d投影矩阵
        SVD或OLS方法求解
    """
    n_points3d = points3d.shape[0]
    n_points2d = points2d.shape[0]
    if n_points3d != n_points2d:
        raise IndexError
    else:
        n_points = n_points3d

    # format equation: Am = b
    A = np.zeros((2 * n_points, 11))
    b = np.zeros( 2 * n_points ) 
    for idx_point in range(n_points):
        point3d = points3d[idx_point]
        point2d = points2d[idx_point]

        x = point3d[0]
        y = point3d[1]
        z = point3d[2]
        u = point2d[0]
        v = point2d[1]

        #debug 
        # print("x: {:3f}, y: {:3f}, z: {:3f}, u: {:3f}, v: {:3f}".format(x,y,z,u,v))

        A[idx_point*2    , :] = np.array([x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z])
        A[idx_point*2 + 1, :] = np.array([0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z])
        b[idx_point*2       ] = u
        b[idx_point*2 + 1   ] = v
    #debug print(A, "\n", b)

    if  method == "ols":
        M = np.eye(4)
        m = np.linalg.lstsq(A, b, rcond=None)[0]
        M[:3, :] = np.reshape(np.append(m, 1), (3, 4))
        return M

    elif method == "svd":
        N = np.eye(4)
        # format equation: Cn = 0
        C = np.hstack((A, -b.reshape((n_points * 2, 1))))
        _, _, VT = np.linalg.svd(C)
        n = VT[-1, :]
        N[:3, :] = n.reshape((3, 4))
        return N
    else:
        raise TypeError

def decompose_projection_mat(mat_projection: np.ndarray):
    """
        分解投影矩阵
        公式法, 旋转矩阵不一定保证正交
    """
    M_ = mat_projection
    m34 = 1 / np.linalg.norm(M_[2, :3])
    M = M_ * m34
    
    m1 = M[0, :3]
    m2 = M[1, :3]
    m3 = M[2, :3]


    fx = np.linalg.norm(np.cross(m1, m3))
    fy = np.linalg.norm(np.cross(m2, m3))

    cx = np.dot(m1, m3)
    cy = np.dot(m2, m3)

    r1 = (m1 - cx*m3) / fx
    r2 = (m2 - cy*m3) / fy
    r3 = m3

    t1 = (M[0, 3] - cx*M[2, 3]) / fx
    t2 = (M[1, 3] - cy*M[2, 3]) / fy
    t3 = M[2, 3]

    mat_intrin = np.array([
            [fx,  0, cx, 0],
            [ 0, fy, cy, 0], 
            [ 0,  0,  1, 0],
            [ 0,  0,  0, 1]
        ])
    mat_extrin = np.eye(4)
    mat_extrin[0, :3] = r1
    mat_extrin[1, :3] = r2
    mat_extrin[2, :3] = r3
    mat_extrin[0,  3] = t1
    mat_extrin[1,  3] = t2
    mat_extrin[2,  3] = t3

    return [mat_intrin, mat_extrin]

def decompose_projection_mat_by_rq(mat_projection: np.ndarray):
    """
        RQ分解投影矩阵,旋转矩阵正交, 但内参skew因子不一定为0
    """
    M = mat_projection

    mat_intrin = np.eye(4)
    mat_extrin = np.eye(4)

    I = np.eye(3)
    P = np.flip(I, 1)
    A = M[:3, :3]
    _A = P @ A
    _Q, _R = np.linalg.qr(_A.T)
    R = P @ _R.T @ P
    Q = P @ _Q.T
    # check
    # print(R @ Q - A < 1E-10)
    
    mat_intrin[:3, :3] = R 
    mat_extrin[:3, :3] = Q 
    mat_extrin[:3,  3] = np.linalg.inv(R) @ M[:3, 3]
    return [mat_intrin, mat_extrin]

def project_points3d_to_2d(rtvec: np.ndarray, mat_projection: np.ndarray, points3d: np.ndarray)-> np.ndarray:
    """
        将3d点投影至2d
    """
    P = np.hstack((points3d, np.ones((points3d.shape[0], 1)))).T
    M = mat_projection

    rvec = rtvec[:3]
    tvec = rtvec[3:]
    #rvec[0] = 0
    #rvec[2] = 0

    T = t_to_T(tvec)
    R = r_to_R(rvec)

    V = T @ R

    points3d_ = (M @ V @ P)
    #points3d_ = points3d_ / points3d_[2]
    points2d = points3d_[:2, :] / points3d_[2]
    points2d = points2d.T
    return points2d

def get_residual(rtvec: np.ndarray, **kwargs_of_func_objective):
    """
        计算残差 \n
        rtvec, {mat_projection, points3d, points2d_object}
    """
    points2d_object = kwargs_of_func_objective["points2d_object"]

    points2d_projected = project_points3d_to_2d(rtvec, mat_projection=kwargs_of_func_objective["mat_projection"], points3d=kwargs_of_func_objective["points3d"])
    residual = points2d_object - points2d_projected
    return residual

# def get_residual_multi(rtvec: np.ndarray, args):
#     """
#     rtvec, [mats_projection_of_n_cams, points3d_for_all_cams, points2d_object_n_cams]
#     """
#     Ms = args[0]
#     points3d = args[1]
#     points2d_object_n_cams = args[2]

#     n_cams = len(points2d_object_n_cams)
#     n_points = points3d.shape[0]

#     residual_multi = np.zeros((2, n_points))
#     avg_loss = 0
#     for i in range(n_cams):
#         residual = get_residual(rtvec, [Ms[i], points3d, points2d_object_n_cams[i]])
#         avg_loss += np.average(loss)
#         loss_multi_cams[i] = loss
#     return residual

def get_reprojection_error(rtvec: np.ndarray, **kwargs_of_func_objective):
    """
        计算投影误差 \n
        rtvec, {mat_projection, points3d, points2d_object}
    """
    delta = get_residual(rtvec, **kwargs_of_func_objective)
    loss = np.sqrt(np.diag(delta @ delta.T)) # L2
    return loss

def get_reprojection_error_multi(rtvec: np.ndarray, **kwargs_of_func_objective_multi):
    """
        计算多相机投影误差 \n
        rtvec, {mats_projection_of_n_cams, points3d_of_n_cams, points2d_of_n_cams}
    """
    mats_projection_of_n_cams = kwargs_of_func_objective_multi["mats_projection_of_n_cams"]
    points3d_n_cams           = kwargs_of_func_objective_multi["points3d_of_n_cams"]
    points2d_object_n_cams    = kwargs_of_func_objective_multi["points2d_of_n_cams"]

    n_cams = len(points2d_object_n_cams)
    n_points = points3d_n_cams[0].shape[0]

    loss_multi_cams = np.zeros((n_cams, n_points))
    avg_loss = 0
    for i_cam in range(n_cams):
        kwargs_single = EasyDict({})
        kwargs_single.mat_projection  = mats_projection_of_n_cams[i_cam]
        kwargs_single.points3d        = points3d_n_cams[i_cam]
        kwargs_single.points2d_object = points2d_object_n_cams[i_cam]
        loss = get_reprojection_error(rtvec, **kwargs_single)
        #print("cam", i, ":\t", loss)
        avg_loss += np.average(loss)
        loss_multi_cams[i_cam] = loss
    #print(np.average(loss_multi_cams))
    return np.average(loss_multi_cams)

def get_jacobian_matrix(params, func_objective, **kwargs_of_func_objective):
    """
        计算jacobian矩阵, 数值微分法 \n
        params, func_objective, {mat_projection, points3d, points2d_object}
    """
    delta = 1E-6
    n_prams = params.shape[0]
    J = np.zeros(n_prams)
    for [idx_parm, param] in enumerate(params):
        params_delta_p = params.copy()
        params_delta_n = params.copy()

        params_delta_p[idx_parm] = param + delta
        params_delta_n[idx_parm] = param - delta

        loss_delta_p = func_objective(params_delta_p, **kwargs_of_func_objective)
        loss_delta_n = func_objective(params_delta_n, **kwargs_of_func_objective)

        dl_of_dp = (loss_delta_p - loss_delta_n) / (2 * delta)
        J[idx_parm] = dl_of_dp
    return J

def get_jacobian_matrix_parallel(params, func_objective, **kwargs_of_func_objective):
    """
        params, func_objective,  args_of_func_objective:[mats_projection_of_n_cams, points3d_for_all_cams, points2d_object_n_cams]
    """
    delta = 1E-8
    n_prams = params.shape[0]
    #n_cameras = len(args[1])
    J = np.zeros((n_prams))
    for [idx_parm, param] in enumerate(params):
        params_delta_p = params.copy()
        params_delta_n = params.copy()

        params_delta_p[idx_parm] = param + delta
        params_delta_n[idx_parm] = param - delta

        loss_delta_p = func_objective(params_delta_p, **kwargs_of_func_objective)
        loss_delta_n = func_objective(params_delta_n, **kwargs_of_func_objective)

        dl_of_dp = (loss_delta_p - loss_delta_n) / (2 * delta)
        J[idx_parm] = dl_of_dp
    return J

def rotation2d(theta):
    R = np.eye(3)
    R[:2, :2] = np.array([
        [ np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])
    return R