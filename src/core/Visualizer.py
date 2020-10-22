
import glob
import os
from typing import *

import cv2
import numpy as np

from . import geometry as geo

def to_plot(point2d, is_homo=False) -> np.ndarray:
    if is_homo:
        p2d = tuple(np.round(point2d[:2]).flatten().astype(np.int).tolist())
    else:
        p2d = tuple(np.round(point2d).flatten().astype(np.int).tolist())
    
    return p2d

class Visualizer(object):
    def __init__(self):
        return

    def draw(self, mode: str="", **kwargs):
        """
        When calib:
            img = kwargs["img"]
            points2d = kwargs["points2d"]
            points3d = kwargs["points3d"]
            camera_pars = kwargs["camera_pars"]
        When solve:
            img = kwargs["img"]
            points2d = kwargs["points2d"]
            points3d = kwargs["points3d"]
            rtvec = kwargs["rtvec"]
            camera_pars = kwargs["camera_pars"]
        """
        self.mode = mode
        if   self.mode == "calib":
            img = kwargs["img"]
            points2d = kwargs["points2d"]
            points3d = kwargs["points3d"]
            camera_pars = kwargs["camera_pars"]

            self.draw_points2d(img, points2d)
            self.draw_cube2d(img, points2d)
            self.draw_eval3d(img, points3d, camera_pars)
            self.draw_axis3d(img, camera_pars)
        elif self.mode == "solve":
            img = kwargs["img"]
            points2d = kwargs["points2d"]
            points3d = kwargs["points3d"]
            rtvec = kwargs["rtvec"]
            camera_pars = kwargs["camera_pars"]

            self.draw_points2d(img, points2d)
            self.draw_backbone3d(img, points3d, rtvec, camera_pars)
            self.draw_axis3d(img, camera_pars)
        return
    
    def draw_points2d(self, img, points2d_chosen, radius=5, color: Tuple[int]=(0, 127, 0)):
        for point2d in points2d_chosen:
            cv2.circle(img, to_plot(point2d), radius, color, 1, 0)
        return

    def draw_points2d_with_texts(self, img, points2d_chosen, radius=5, color: Tuple[int]=(0, 127, 255)):
        for [i_point, point2d] in enumerate(points2d_chosen):
            cv2.circle(img, to_plot(point2d), radius, color, 1, 0)
            off_set = 5
            text = "{}".format(i_point + 1)
            cv2.putText(img, text, to_plot(point2d + off_set), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color=(0,0,255))
        return
    
    def draw_points3d(self, 
            img: np.ndarray, 
            points3d: np.ndarray, 
            rtvec: np.ndarray, camera_pars: Dict, 
            radius=1,
            color: Tuple[int]=(255, 127, 0)
        ):

        M = camera_pars["intrin"] @ camera_pars["extrin"]
        points2d = geo.project_points3d_to_2d(rtvec, M, points3d)
        self.draw_points2d(img, points2d, radius=radius, color=color)
        return
        
    def draw_points3d_with_texts(self, img, points3d, rtvec, camera_pars, radius=1, color: Tuple[int]=(255, 127, 0)):
        self.draw_points3d(img, points3d, rtvec, camera_pars)
        M = camera_pars["intrin"] @ camera_pars["extrin"]
        points2d = geo.project_points3d_to_2d(rtvec, M, points3d)
        self.draw_points2d(img, points2d, radius=radius, color=color)
        n_points = points2d.shape[0]
        for i_point in range(n_points):
            point2d = points2d[i_point]
            off_set = 5
            text = "{}".format(i_point + 1)
            cv2.putText(img, text, to_plot(point2d + off_set), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color=(0,0,255))
        return

    def draw_backbone3d(self, 
            img: np.ndarray, 
            points3d_backbone: np.ndarray, 
            rtvec: np.ndarray, camera_pars, 
            color: Tuple[int]=(255, 255, 128), 
            width_line = 1
        ):

        M = camera_pars["intrin"] @ camera_pars["extrin"]
        points2d = geo.project_points3d_to_2d(rtvec, M, points3d_backbone)
        n_points = points2d.shape[0]
        if n_points >= 2:
            cv2.line(img, to_plot(points2d[0]), to_plot(points2d[1]), color, width_line)
        if n_points >= 3:
            cv2.line(img, to_plot(points2d[0]), to_plot(points2d[2]), color, width_line)
        if n_points >= 4:
            cv2.line(img, to_plot(points2d[0]), to_plot(points2d[3]), color, width_line)
        if n_points >= 5:
            cv2.line(img, to_plot(points2d[0]), to_plot(points2d[4]), color, width_line)

            cv2.line(img, to_plot(points2d[1]), to_plot(points2d[2]), color, width_line)
            cv2.line(img, to_plot(points2d[3]), to_plot(points2d[4]), color, width_line)

            cv2.line(img, to_plot(points2d[1]), to_plot(points2d[3]), color, width_line)
            cv2.line(img, to_plot(points2d[1]), to_plot(points2d[4]), color, width_line)
            cv2.line(img, to_plot(points2d[2]), to_plot(points2d[3]), color, width_line)
            cv2.line(img, to_plot(points2d[2]), to_plot(points2d[4]), color, width_line)
        return

    def draw_cube2d(self, img: np.ndarray, points2d_cube: np.ndarray, color: Tuple[int]=(0, 0, 255), width_line: int=1):
        pairs = []
        n_points = points2d_cube.shape[0]
        if   n_points == 8:
            pairs = [
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 8],
                [0, 4], [1, 5], [2, 6], [3, 7],
                ]
        elif n_points == 7:
            pairs = [
                [0, 1], [1, 2], [2, 0], 
                [3, 4], [4, 5], [5, 6], [6, 3],
                [0, 3], [1, 4], [2, 5], 
                ]
        elif n_points == 6:
            pairs = [
                [0, 1],  
                [2, 3], [3, 4], [4, 5], [5, 2],
                [0, 2], [1, 3],  
                ]
        for pair in pairs:
            [i, j] = pair
            cv2.line(
                img, 
                to_plot(points2d_cube[i]),
                to_plot(points2d_cube[j]),
                color,
                width_line)
        return
 
    def draw_eval3d(self, img: np.ndarray, points3d_cube: np.ndarray, camera_pars, color: Tuple[int]=(0, 0, 127), width_line: int=1):
        a =  np.pi 
        addon = np.array([
            [ np.cos(a), np.sin(a), 0, 0],
            [-np.sin(a), np.cos(a), 0, 0],
            [         0,         0, 1, 0],
            [         0,         0, 0, 1]
        ])
        addon2 = np.array([
            [ 1, 0, 0, 0.1],
            [ 0, 1, 0, 0],
            [ 0, 0, 1, 0],
            [ 0, 0, 0, 1]
        ])
        
        M = camera_pars["intrin"] @ camera_pars["extrin"]
        M = M @ addon2 @ addon 
        points2d_cube = geo.project_points3d_to_2d(np.zeros(6), M, points3d_cube)
         
        self.draw_cube2d(img, points2d_cube, color, width_line)
        return
   
    def draw_axis3d(self, img, camera_pars):
        M = camera_pars["intrin"] @ camera_pars["extrin"]
        p2ds = geo.project_points3d_to_2d(np.zeros(6), M, np.array([[0,0,0], [0.1,0,0], [0,0.1,0], [0,0,0.1]]))
        cv2.line(img, to_plot(p2ds[0]), to_plot(p2ds[1]), (0, 0, 255), 1)
        cv2.line(img, to_plot(p2ds[0]), to_plot(p2ds[2]), (0, 255, 0), 1)
        cv2.line(img, to_plot(p2ds[0]), to_plot(p2ds[3]), (255, 0, 0), 1)
        return
        
    def draw_triangle2d(self, img, points2d_tri, color):
        p2ds = []
        for i in [0, 1, 2]:
            p2d = to_plot(points2d_tri[i])
            p2ds.append(p2d)

        cv2.line(img, p2ds[0], p2ds[1], color, 1)
        cv2.line(img, p2ds[1], p2ds[2], color, 1)
        cv2.line(img, p2ds[2], p2ds[0], color, 1)
        return

    def draw_model3d(self, img, model, rtvec, camera_pars, color=(0, 255, 0)):
        M = camera_pars["intrin"] @ camera_pars["extrin"]
        points3d_model = []
        points2d_model = []
        for tri in (model):
            points3d_tri = []
            points2d_tri = []
            for point3d in tri:
                p3d = np.array([[point3d[0], point3d[1], point3d[2]]]) / 1000
                points3d_tri.append(p3d)
                p2d = geo.project_points3d_to_2d(
                    rtvec, 
                    M,
                    p3d
                )
                points2d_tri.append(p2d)
            points3d_model.append(points3d_tri)
            points2d_model.append(points2d_tri)
            self.draw_triangle2d(img, points2d_tri, color)
        return

    def draw_model3d_mask(self, img, rtvec, camera_pars, model, color=(255, 255, 255)):
        M = camera_pars["intrin"] @ camera_pars["extrin"]
        points2d_n_tris = []
        for tri in (model):
            points2d_tri = []
            for point3d in tri:
                p3d = np.array([[point3d[0], point3d[1], point3d[2]]]) / 1000
                p2d = geo.project_points3d_to_2d(
                    rtvec, 
                    M,
                    p3d
                )
                points2d_tri.append(to_plot(p2d))
            points2d_n_tris.append(points2d_tri) 
            cv2.fillPoly(img, np.array([points2d_tri]), color)
            # cv2.imshow("", img)
            # cv2.waitKey(0)   
        return
    
# if __name__ == "__main__":
#     import FileIO
#     import json
#     pth_model = "/home/veily/LiGan/Pose6dSolver/test/9-14/输入/测量/模型/圆柱_椎.STL"
#     fio = FileIO.FileIO("solve")
#     model = fio.load_model_from_stl_binary(pth_model)
#     keypts3d = fio.load_points3d("/home/veily/LiGan/Pose6dSolver/test/9-14/输入/测量/模型/关键点.txt")
    
#     Ms = []
#     for pth_cam in [
#         "/home/veily/LiGan/Pose6dSolver/test/9-10/output/标定/1/datas/camera_pars.json", 
#         "/home/veily/LiGan/Pose6dSolver/test/9-10/output/标定/2/datas/camera_pars.json"
#     ]:
#         with open(pth_cam) as f:
#             camera = json.load(f)
#         I = np.array(camera["intrin"])
#         E = np.array(camera["extrin"])
#         M = I @ E 
#         cam = fio.load_camera_pars(pth_cam)
#         Ms.append(cam)

#     rtvec = np.zeros(6)
# #     rtvec = np.array([
# #         0, 5*np.pi/180, 0, 
# #         0, 0, 0
# #     ])
# #     rtvec = np.array([ 2.278232413567098380e+00 ,9.393448721296800141e-02, -1.929144611022573785e-01, -5.345969458186864420e-01, 9.962088787458184269e-02 ,-5.773752920191686094e-02
# # ])
#     pths_imgs=["/home/veily/LiGan/Pose6dSolver/test/9-10/input/标定/1/000_1.jpg", "/home/veily/LiGan/Pose6dSolver/test/9-10/input/标定/2/000_2.jpg"]

#     img1 = cv2.imread(pths_imgs[0])
#     img2 = cv2.imread(pths_imgs[1])
#     cv2.namedWindow(str(1), cv2.WINDOW_FREERATIO)
#     cv2.namedWindow(str(2), cv2.WINDOW_FREERATIO)
#     M1 = Ms[0]
#     M2 = Ms[1]
#     vis = Visualizer("solve")
#     vis.draw_axis3d(img1, M1)
#     vis.draw_axis3d(img2, M2)
#     import matplotlib.pyplot as plt

#     dvec = np.array([ 0.  ,  0.  ,  0.  , 0,  0 , 0 ])
#     while 1:
#         img1 = cv2.imread(pths_imgs[0])
#         img2 = cv2.imread(pths_imgs[1])
#         vis.draw_axis3d(img1, M1)
#         vis.draw_axis3d(img2, M2)
#         # cv2.imshow(str(1), img1)
#         # cv2.imshow(str(2), img2)
#         vis.draw_model3d(img1, rtvec+dvec, M1, model)
#         vis.draw_model3d(img2, rtvec+dvec, M2, model)#[:100])
#         vis.draw_backbone3d(img1, keypts3d, rtvec, M1)
#         vis.draw_backbone3d(img2, keypts3d, rtvec, M2)
#         key=cv2.waitKey(10)
#         cv2.imshow("1", img1)
#         cv2.imshow("2", img2)

#         if key == ord("c"):
#             cv2.imwrite("/home/veily/LiGan/Pose6dSolver/test/9-14/new_method/init_1.png", img1)
#             cv2.imwrite("/home/veily/LiGan/Pose6dSolver/test/9-14/new_method/init_2.png", img2)
#             # vis.draw_model3d(img1, rtvec+dvec, M1, model)#[:100])
#             # vis.draw_model3d(img2, rtvec+dvec, M2, model)#[:100])
#             # cv2.imshow(str(1), img1)
#             # cv2.imshow(str(2), img2)
#             # key = cv2.waitKey(0)

#         if key == ord("q"):
#             dvec += np.array([0,0,0,  0.01,0,0])
#             cv2.waitKey(1)
#         if key == ord("w"):
#             dvec += np.array([0,0,0, -0.01,0,0])
#         if key == ord("a"):
#             dvec += np.array([0,0,0,  0,0.01,0])
#             cv2.waitKey(1)
#         if key == ord("s"):
#             dvec += np.array([0,0,0, 0,-0.01,0])
#         if key == ord("z"):
#             dvec += np.array([0,0,0,  0,0,0.01])
#             cv2.waitKey(1)
#         if key == ord("x"):
#             dvec += np.array([0,0,0, 0,0,-0.01])
#             cv2.waitKey(1)
#         if key == ord("p"):
#             break
        
#     print()
