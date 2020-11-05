
import glob
import os
import sys
from typing import *

import cv2
import numpy as np
from easydict import EasyDict

sys.path.append("..")
from core import geometry as geo

def to_plot(point2d, is_homo=False) -> np.ndarray:
    if is_homo:
        p2d = tuple(np.round(point2d[:2]).flatten().astype(np.int).tolist())
    else:
        p2d = tuple(np.round(point2d).flatten().astype(np.int).tolist())
    
    return p2d
    
def draw_points2d(img, points2d_chosen, radius=5, color: Tuple[int]=(0, 127, 0)):
    for point2d in points2d_chosen:
        cv2.circle(img, to_plot(point2d), radius, color, 1, 0)
    return

def draw_points2d_with_texts(img, points2d_chosen, radius=5, color: Tuple[int]=(0, 127, 255)):
    for [i_point, point2d] in enumerate(points2d_chosen):
        cv2.circle(img, to_plot(point2d), radius, color, 1, 0)
        off_set = 5
        text = "{}".format(i_point + 1)
        cv2.putText(img, text, to_plot(point2d + off_set), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color=(0,0,255))
    return

def draw_points3d(
        img: np.ndarray, 
        points3d: np.ndarray, 
        rtvec: np.ndarray, camera_pars: Dict, 
        radius=1,
        color: Tuple[int]=(255, 127, 0)
    ):

    M = camera_pars["intrin"] @ camera_pars["extrin"]
    points2d = geo.project_points3d_to_2d(rtvec, M, points3d)
    draw_points2d(img, points2d, radius=radius, color=color)
    return
    
def draw_points3d_with_texts(img, points3d, rtvec, camera_pars, radius=1, color: Tuple[int]=(255, 127, 0)):
    draw_points3d(img, points3d, rtvec, camera_pars)
    M = camera_pars["intrin"] @ camera_pars["extrin"]
    points2d = geo.project_points3d_to_2d(rtvec, M, points3d)
    draw_points2d(img, points2d, radius=radius, color=color)
    n_points = points2d.shape[0]
    for i_point in range(n_points):
        point2d = points2d[i_point]
        off_set = 5
        text = "{}".format(i_point + 1)
        cv2.putText(img, text, to_plot(point2d + off_set), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color=(0,0,255))
    return

def draw_backbone3d(
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

def draw_axis3d(img, camera_pars: Dict, rtvec: np.ndarray=np.zeros(6), unit_length: float=0.1, width_line: int=1):
    M = camera_pars["intrin"] @ camera_pars["extrin"]
    p2ds = geo.project_points3d_to_2d(rtvec, M, np.array([[0, 0, 0], [unit_length, 0, 0], [0, unit_length, 0], [0, 0, unit_length]]))
    cv2.line(img, to_plot(p2ds[0]), to_plot(p2ds[1]), (0, 0, 255), width_line)
    cv2.line(img, to_plot(p2ds[0]), to_plot(p2ds[2]), (0, 255, 0), width_line)
    cv2.line(img, to_plot(p2ds[0]), to_plot(p2ds[3]), (255, 0, 0), width_line)
    return
    
def draw_triangle2d(img, points2d_tri, color):
    p2ds = []
    for i in [0, 1, 2]:
        p2d = to_plot(points2d_tri[i])
        p2ds.append(p2d)

    cv2.line(img, p2ds[0], p2ds[1], color, 1)
    cv2.line(img, p2ds[1], p2ds[2], color, 1)
    cv2.line(img, p2ds[2], p2ds[0], color, 1)
    return

def draw_model3d(img, model, rtvec, camera_pars, color=(0, 255, 0)):
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
        draw_triangle2d(img, points2d_tri, color)
    return

def draw_model3d_mask(img, rtvec, camera_pars, model, color=(255, 255, 255)):
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

class Visualizer(object):
    def __init__(self):
        self.props = EasyDict({
                "is_draw_axis_world": True,
                "is_draw_axis_self" : True,
                "is_draw_points2d"  : True,
                "is_draw_points3d"  : True,
                "is_draw_model"     : True,
                "is_draw_backbone"  : True,
            })
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
        img         = kwargs["img"]
        points2d    = kwargs["points2d"]
        points3d    = kwargs["points3d"]
        camera_pars = kwargs["camera_pars"]
        rtvec       = kwargs["rtvec"]
        model       = kwargs["model"]

        if self.props.is_draw_model:
            draw_model3d(img, model, rtvec, camera_pars)
        if self.props.is_draw_backbone:
            draw_backbone3d(img, points3d, rtvec, camera_pars)
        if self.props.is_draw_points2d:
            draw_points2d_with_texts(img, points2d)
        if self.props.is_draw_points3d:
            draw_points3d_with_texts(img, points3d, rtvec, camera_pars)
        if self.props.is_draw_axis_world:
            draw_axis3d(img, camera_pars)
        if self.props.is_draw_axis_self:
            draw_axis3d(img, camera_pars, rtvec)
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
