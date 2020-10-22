
import json
import math

import cv2
import numpy as np



def R_to_r(R: np.ndarray)-> np.ndarray:
    R_ = R[:3, :3]
    rvec = cv2.Rodrigues(R_)[0].flatten()
    return rvec

def r_to_R(rvec: np.ndarray)-> np.ndarray:
    R = np.eye(4)
    R_3x3 = cv2.Rodrigues(rvec)[0]
    R[:3,  :3] = R_3x3
    return R

def T_to_t(T: np.ndarray)-> np.ndarray:
    tvec = T[:3, 3]
    return tvec

def t_to_T(tvec: np.ndarray)-> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = tvec
    return T

def rtvec_degree2rad(rtvec_degree: np.ndarray) -> np.ndarray:
    rtvec_rad = rtvec_degree.copy()
    rtvec_rad[:3] = np.pi * (rtvec_rad[:3] / 180)
    return rtvec_rad
    
def rtvec_rad2degree(rtvec_rad: np.ndarray) -> np.ndarray:
    rtvec_degree = rtvec_rad.copy()
    rtvec_degree[:3] = 180 * (rtvec_degree[:3] / np.pi)
    return rtvec_degree

def solve_projection_mat_3d_to_2d(points3d: np.ndarray, points2d: np.ndarray, method="svd")-> np.ndarray:
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

def decomposition_intrin_extrin_from_projection_mat(mat_projection: np.ndarray):
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

def get_residual(rtvec: np.ndarray, args):
    """
    rtvec, [mat_projection, points3d, points2d_object]
    """
    M = args[0]
    points3d = args[1]
    points2d_object = args[2]
    points2d_projected = project_points3d_to_2d(rtvec, M, points3d)
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

def get_reprojection_error(rtvec: np.ndarray, args):
    """
    rtvec, [mat_projection, points3d, points2d_object]
    """
    delta = get_residual(rtvec, args)
    loss = np.sqrt(np.diag(delta @ delta.T)) # L2
    return loss

def get_reprojection_error_multi(rtvec: np.ndarray, args):
    """
    rtvec, [mats_projection_of_n_cams, points3d_for_all_cams, points2d_object_n_cams]
    """
    Ms = args[0]
    points3d = args[1]
    points2d_object_n_cams = args[2]

    n_cams = len(points2d_object_n_cams)
    n_points = points3d.shape[0]

    loss_multi_cams = np.zeros((2, n_points))
    avg_loss = 0
    for i in range(n_cams):
        loss = get_reprojection_error(rtvec, [Ms[i], points3d, points2d_object_n_cams[i]])
        #print("cam", i, ":\t", loss)
        avg_loss += np.average(loss)
        loss_multi_cams[i] = loss
    #print(np.average(loss_multi_cams))
    return np.average(loss_multi_cams)

def get_jacobian_matrix(params, func_objective, args_of_func_objective):
    """
    params, func_objective, args_of_func_objective:[mat_projection, points3d, points2d_object]
    """
    delta = 1E-6
    n_prams = params.shape[0]
    n_objects = np.shape(args_of_func_objective[-1])[0]
    J = np.zeros(n_prams)
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

def get_jacobian_matrix_multi(params, func_objective, args):
    """
    params, func_objective,  args_of_func_objective:[mats_projection_of_n_cams, points3d_for_all_cams, points2d_object_n_cams]
    """
    delta = 1E-8
    n_prams = params.shape[0]
    n_points = np.shape(args[-1][0])[0]
    #n_cameras = len(args[1])
    J = np.zeros((n_prams))
    for [idx_parm, param] in enumerate(params):
        params_delta_p = params.copy()
        params_delta_n = params.copy()
        params_delta_p[idx_parm] = param + delta
        params_delta_n[idx_parm] = param - delta

        loss_delta_p = func_objective(params_delta_p, args)
        loss_delta_n = func_objective(params_delta_n, args)

        dl_of_dp = (loss_delta_p - loss_delta_n) / (2 * delta)
        J[idx_parm] = dl_of_dp
    return J
