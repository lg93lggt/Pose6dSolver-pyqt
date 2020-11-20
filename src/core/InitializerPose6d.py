

import numpy as np
import sys
import cv2
from easydict import EasyDict
from numpy.linalg.linalg import solve

sys.path.append("./src")
from core import ParticleSwarmOptimization
from core import geometry as geo

class InitializerPose6d(object):
    def __init__(self, method="PSO"):
        self.method = method
        self.theta0 = np.zeros(6)
        return

    def run_by_epnp(self, **kwargs_epnp):
        """
            {
                cameras            
                points3d_of_n_cams 
                points2d_of_n_cams 
            }
        """
        print("\nEPnP: ")
        cams               = kwargs_epnp["cameras"]            if ("cameras" in kwargs_epnp.keys())            else None
        points3d_of_n_cams = kwargs_epnp["points3d_of_n_cams"] if ("points3d_of_n_cams" in kwargs_epnp.keys()) else None
        points2d_of_n_cams = kwargs_epnp["points2d_of_n_cams"] if ("points2d_of_n_cams" in kwargs_epnp.keys()) else None

        if (cams is None) or (points3d_of_n_cams is None) or (points2d_of_n_cams is None):
            raise ValueError("错误: 输入参数名称错误.")

        mats_projection_of_n_cams = []
        for [i_cam, cam] in enumerate(cams):
            mats_projection_of_n_cams.append(cam.intrin @ cam.extrin)

        n_cams     = len(cams)
        theta_all  = np.zeros((n_cams, 6))
        losses_all = np.zeros(n_cams)
        for [i_cam, cam] in enumerate(cams):
            print("解算 {}/{}: ".format(i_cam + 1, n_cams))
            dist_coeffs = cam.dist_coeffs if ("dist_coeffs" in cam.keys()) else np.zeros(5)
            pts3d  = points3d_of_n_cams[i_cam].reshape((-1, 1, 3))
            pts2d  = points2d_of_n_cams[i_cam].reshape((-1, 1, 2))
            intrin = cam.intrin[:3, :3]
            extrin = cam.extrin
            [ret, rvec, tvec] = cv2.solvePnP(objectPoints=pts3d, imagePoints=pts2d, cameraMatrix=intrin, distCoeffs=dist_coeffs, flags=cv2.SOLVEPNP_P3P)
            pose_obj_to_cam = geo.rtvec_to_rtmat(np.hstack((rvec.T, tvec.T)))
            pose_obj = np.linalg.inv(extrin) @ pose_obj_to_cam 
            theta_tmp = geo.rtmat_to_rtvec(pose_obj)
            theta_all[i_cam] = theta_tmp
            loss = geo.get_reprojection_error_multi(
                theta_tmp, 
                mats_projection_of_n_cams=mats_projection_of_n_cams, 
                points3d_of_n_cams=points3d_of_n_cams, 
                points2d_of_n_cams=points2d_of_n_cams
            )
            losses_all[i_cam] = loss
            print("theta: \n", theta_tmp, "\n", "loss: \n", loss)
        theta0 = theta_all[np.argmin(losses_all)]
        return theta0
        

    def init_pso(self, **kwargs_pso_init):
        """
            {n_pops: 100, n_dims: 6, n_iters: 500, w: 0.99, c1: 2, c2: 2}
        """
        self.pso = ParticleSwarmOptimization.ParticleSwarmOptimization(**kwargs_pso_init)
        return

    def run_by_pso(self, **kwargs_pso_run):
        """
            {cameras, points3d_of_n_cams, points2d_of_n_cams}
        """
        cams = kwargs_pso_run["cameras"] if ("cameras" in kwargs_pso_run.keys()) else None
        points3d_of_n_cams = kwargs_pso_run["points3d_of_n_cams"] if ("points3d_of_n_cams" in kwargs_pso_run.keys()) else None
        points2d_of_n_cams = kwargs_pso_run["points2d_of_n_cams"] if ("points2d_of_n_cams" in kwargs_pso_run.keys()) else None

        if (cams is None) or (points3d_of_n_cams is None) or (points2d_of_n_cams is None):
            raise ValueError("错误: 输入参数名称错误.")

        mats_projection_of_n_cams = []
        for cam in cams:
            mats_projection_of_n_cams.append(cam.intrin @ cam.extrin)

        kwargs_of_func_objective = {
            "mats_projection_of_n_cams": mats_projection_of_n_cams,
            "points3d_of_n_cams": points3d_of_n_cams,
            "points2d_of_n_cams": points2d_of_n_cams
        }

        self.pso.set_objective_func(geo.get_reprojection_error_multi)
        self.pso.set_boundery(
            lower_bound=np.array([-np.pi, -np.pi, -np.pi, -5, -5, -5]),
            upper_bound=np.array([+np.pi, +np.pi, +np.pi, +5, +5, +5])
        )
        self.pso.run(**kwargs_of_func_objective)
        theta0 = self.pso.global_best
        return theta0



if __name__ == "__main__":
    from core import FileIO
    from core import Visualizer
    from core import SolverPoses6d
    solver = SolverPoses6d.SolverPoses6dDLT("Adam", alpha=0.01, beta1=0.9, beta2=0.999)

    fio = FileIO.FileIO()
    sys.path.append("../..")
    fio.load_project_from_filedir("./姿态测量4")
    _, pts3d = fio.load_points3d("solve", 0)
    pts2d_all = []
    pts3d_all = []
    cams_all  = []
    mat_proj  = []
    for i in range(2):
        ret, pts2d = fio.load_points2d("solve", 19, 0, i)
        pts2d_all.append(pts2d.astype(np.float))
        ret, idxs3d = fio.load_indexes3d("solve", 19, 0, i)
        pts3d_all.append(pts3d[idxs3d])
        _, cam = fio.load_camera_pars(i)
        cams_all.append(cam)
        mat_proj.append(cam.intrin @ cam.extrin)
    initer = InitializerPose6d()
    initer.init_pso(n_iters=200, n_pops=100, w=0.618, c1=2, c2=2)
    kwargs = {"cameras": cams_all, "points3d_of_n_cams": pts3d_all, "points2d_of_n_cams": pts2d_all}
    initer.run_by_pso(**kwargs)
     
    x0 = initer.run_by_epnp(cameras=cams_all, points3d_of_n_cams=pts3d_all, points2d_of_n_cams=pts2d_all)
    print(x0, geo.get_reprojection_error_multi(x0, **kwargs))
    print(initer.pso.global_best, geo.get_reprojection_error_multi(initer.pso.global_best, **kwargs))
    # -1.626738 -0.187478 2.437883 0.169671 0.079590 -0.314358

    imgs = [fio.load_image_raw("solve", 19, 0)[1], fio.load_image_raw("solve", 19, 1)[1]]
    _, backbone_lines = fio.load_backbonelines("solve", 0)

    solver.run(initer.pso.global_best, **kwargs)
    x = solver.opt.theta
    for [i_img, img] in enumerate(imgs):
        img_ = img.copy()
        Visualizer.draw_backbone3d(img_, pts3d, backbone_lines, x, cams_all[i_img])
        cv2.namedWindow("pso_"+str(i_img), cv2.WINDOW_NORMAL)
        cv2.imshow("pso_"+str(i_img), img_)
        cv2.waitKey()

    solver.run(x0[0], **kwargs)
    x = solver.opt.theta
    for [i_img, img] in enumerate(imgs):
        img_ = img.copy()
        Visualizer.draw_backbone3d(img_, pts3d, backbone_lines, x, cams_all[i_img])
        cv2.namedWindow("x0_"+str(i_img), cv2.WINDOW_NORMAL)
        cv2.imshow("x0_"+str(i_img), img_)
        cv2.waitKey()

    solver.run(x0[1], **kwargs)
    x = solver.opt.theta
    for [i_img, img] in enumerate(imgs):
        img_ = img.copy()
        Visualizer.draw_backbone3d(img_, pts3d, backbone_lines, x, cams_all[i_img])
        cv2.namedWindow("x1_"+str(i_img), cv2.WINDOW_NORMAL)
        cv2.imshow("x1_"+str(i_img), img_)
        cv2.waitKey()

    solver.run(x_avg, **kwargs)
    x = solver.opt.theta
    for [i_img, img] in enumerate(imgs):
        img_ = img.copy()
        Visualizer.draw_backbone3d(img_, pts3d, backbone_lines, x, cams_all[i_img])
        cv2.namedWindow("x_avg_"+str(i_img), cv2.WINDOW_NORMAL)
        cv2.imshow("x_avg_"+str(i_img), img_)
        cv2.waitKey()

    print()

    