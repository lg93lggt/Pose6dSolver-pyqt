

import math
from operator import ge
import numpy as np
import cv2
from cv2 import aruco 
import json

from Adam import Adam
import geometry as geo

class SolverPoses6d(object):
    def __init__(self) -> None:
        self.opt = Adam(n_iters=5000, alpha=0.001, beta1=(1 - 0.1), beta2=(1 - 0.0001))
        return

    def set_points3d_real_for_all(self, points3d_real_for_all: np.ndarray):
        self.points3d_real_for_all = points3d_real_for_all
        return

    def load_points3d_real_for_all_from_file(self, pth_points3d_real):
        self.pth_points3d_real = pth_points3d_real
        points3d = np.loadtxt(self.pth_points3d_real)
        self.set_points3d_real_for_all(points3d)
        return

    def get_projection_mats_for_n_cams(self):
        self.mats_projections_for_n_cams = []
        for cam in self.cameras:
            mat_intrin = np.array(cam["intrin"]) 
            mat_extrin = np.array(cam["extrin"]) 
            M = mat_intrin @ mat_extrin
            self.mats_projections_for_n_cams.append(M)
        return

    def load_cameras_pars_from_file(self, *pths_cams_pars):
        self.cameras = []
        self.pths_cameras_pars = pths_cams_pars[0]
        self.n_cameras = len(self.pths_cameras_pars)
        for pth_cam_pars in self.pths_cameras_pars:
            with open(pth_cam_pars) as fp:
                camera = json.load(fp)
                self.cameras.append(camera)
        self.get_projection_mats_for_n_cams()
        return

    def set_points2d_for_n_cams(self, *points2d_for_n_cams):
        self.points2d_for_n_cams = points2d_for_n_cams
        return

    def load_points2d_for_n_cams_from_file(self, *pths_points2d_for_n_cams):
        self.pths_points2d_for_n_cams = pths_points2d_for_n_cams[0]
        self.points2d_obj_of_n_cams = []
        for pth in self.pths_points2d_for_n_cams:
            points2d = np.loadtxt(pth)
            self.points2d_obj_of_n_cams.append(points2d)
        return

    def set_images_for_n_cams(self, imgs):
        self.imgs_raw_of_n_cams = []
        self.imgs_of_n_cams = []
        for img in imgs:
            self.imgs_raw_of_n_cams.append(img)    
            self.imgs_of_n_cams.append(img.copy())
        return

    def load_images_for_n_cams_from_file(self, *pths_images_for_n_cams):
        self.pths_images_for_n_cams = pths_images_for_n_cams[0]
        imgs = []
        for pth in self.pths_images_for_n_cams:
            print(pth)
            img = cv2.imread(pth)
            imgs.append(img)
        self.set_images_for_n_cams(imgs)
        return    

    def solve(self, theta0=np.zeros(6)):
        self.opt.set_objective_func(geo.get_reprojection_error_multi)
        self.opt.set_jacobian_func(geo.get_jacobian_matrix_multi)
        [log_loss, log_theta] = self.opt.run(theta0, self.mats_projections_for_n_cams, self.points3d_real_for_all, self.points2d_obj_of_n_cams)
        return [log_loss, log_theta]

    def run(self, theta0=np.zeros(6)):
        [log_loss, log_theta] = self.solve(theta0)
        for i in range(self.n_cameras):
            points2d = geo.project_points3d_to_2d(self.opt.theta, self.mats_projections_for_n_cams[i], self.points3d_real_for_all)
            self.draw_backbone(self.imgs_of_n_cams[i], points2d)
            cv2.imshow(str(i), self.imgs_of_n_cams[i])
            cv2.waitKey(0)
        return self.opt.theta

    def draw_obj_points_for_all(self, imgs, obj_points):
        for i in [0, 1]:
            for j in range(6):
                cv2.circle(imgs[i], (obj_points[i][j, 0], obj_points[i][j, 1]), 2, (0, 0, 128), 1, 1)
        return

    def draw_backbone(self, img, projected_points2d):
        points2d = projected_points2d.astype(np.int)
        line_width = 1
        #cv2.circle(img, tuple(points2d[0, :2]), 2, (255, 255, 128), 1)
        n_points = points2d.shape[0]
        if n_points >= 2:
            cv2.line(img, tuple(points2d[0, :2]), tuple(points2d[1, :2]), (255, 255, 128), line_width)
        if n_points == 3:
            cv2.line(img, tuple(points2d[0, :2]), tuple(points2d[2, :2]), (255, 255, 128), line_width)
        if n_points >= 4:
            cv2.line(img, tuple(points2d[2, :2]), tuple(points2d[3, :2]), (255, 255, 128), line_width)
        if n_points == 5:
            cv2.line(img, tuple(points2d[0, :2]), tuple(points2d[4, :2]), (255, 255, 128), line_width)
        if n_points == 6:
            cv2.line(img, tuple(points2d[4, :2]), tuple(points2d[5, :2]), (255, 255, 128), line_width)

        return

    





if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("输入模型3D关键点文件路径, 各相机参数文件, 各相机图像路径, 各相机2D关键点文件路径, 输出文件夹.")
    parser.add_argument(
        "-icams",
        type=str, 
        nargs='+',
        help="输入相机1参数文件路径",
        required=True
    )
    parser.add_argument(
        "-iimgs",
        type=str, 
        nargs='+',
        help="输入图像路径",
        required=True
    )
    parser.add_argument(
        "-ipts2d", 
        type=str, 
        help="输入3D关键点文件路径",
        nargs='+',
        required=True
    )
    parser.add_argument(
        "-ipts3d", 
        type=str, 
        help="输入2D关键点文件路径",
        required=True
    )
    parser.add_argument(
        "-o", 
        type=str, 
        help="输出文件夹",
        default="./",
        #required=True
    )
    parser.add_argument(
        "-simg", "--save_img", 
        default=False, 
        type=bool, 
        help="是否保存优化过程图片"
    )
    args = parser.parse_args()
    print(args)
    
    # solver.set_points3d_real_for_all(
    #     np.array([
    #         [ 0.  ,  0.  , 0.  ],
    #         [ 0.  ,  0.  , 0.05],
    #         [ 0.  ,  0.01, 0.  ],
    #         [ 0.  , -0.01, 0.  ],
    #         [ 0.01,  0.  , 0.  ],
    #         [-0.01,  0.  , 0.  ]
    #     ]) 
    # )
    solver = SolverPoses6d()
    solver.load_cameras_pars_from_file(args.icams)
    solver.load_images_for_n_cams_from_file(args.iimgs)
    solver.load_points2d_for_n_cams_from_file(args.ipts2d)
    solver.load_points3d_real_for_all_from_file(args.ipts3d)
    x = solver.run()
    degree = x[:3] / np.pi * 180
    print("theta=", x)
    print("degree=", degree)
    print()
    

        