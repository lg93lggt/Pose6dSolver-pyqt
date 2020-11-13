
import numpy as np
import cv2
import json
import sys
sys.path.append("..")
from core import  FileIO, geometry, Visualizer

fio = FileIO.FileIO()
fio.load_project_from_filedir("../../姿态测量")

img0 = fio.load_image_raw("solve", 1, 0)
img1 = fio.load_image_raw("solve", 1, 1)

cam0 = fio.load_camera_pars(0)
cam1 = fio.load_camera_pars(1)

p3d = fio.load_points3d("solve", 0)[:4]

p2d0 = np.array([   
    [106,  83],
    [ 69, 189],
    [ 81, 205],
    [ 60, 211],
    [ 46, 194]
], dtype=float)[:4]
p2d1 = np.array([    
    [275, 108], 
    [153, 188],
    [164, 202],
    [156, 203],
    [146, 190]
], dtype=float)[:4]

_, r0, t0 = cv2.solvePnP(np.ascontiguousarray(p3d[:,:3]).reshape((-1, 1, 3)), np.ascontiguousarray(p2d0[:, :2]).reshape((-1, 1, 2)), cam0.intrin[:3, :3], np.zeros(5), flags=cv2.SOLVEPNP_EPNP)
_, r1, t1 = cv2.solvePnP(np.ascontiguousarray(p3d[:,:3]).reshape((-1, 1, 3)), np.ascontiguousarray(p2d1[:, :2]).reshape((-1, 1, 2)), cam1.intrin[:3, :3], np.zeros(5), flags=cv2.SOLVEPNP_EPNP)
rt0 = np.hstack([r0.flatten(), t0.flatten()])
rt1 = np.hstack([r1.flatten(), t1.flatten()])

M0 =  geometry.t_to_T(t0) @ geometry.r_to_R(r0) 
M1 =  geometry.t_to_T(t1) @ geometry.r_to_R(r1) 
Visualizer.draw_points2d(img0, p2d0)
Visualizer.draw_points2d(img1, p2d1)
cam0.extrin = np.eye(4)
cam1.extrin = np.eye(4)
Visualizer.draw_points3d(img0, p3d, rt0, cam0)
Visualizer.draw_points3d(img1, p3d, rt1, cam1)

cv2.namedWindow("0", flags=cv2.WINDOW_FREERATIO)
cv2.namedWindow("1", flags=cv2.WINDOW_FREERATIO)
cv2.imshow("0", img0)
cv2.imshow("1", img1)
cv2.waitKey()
print()

