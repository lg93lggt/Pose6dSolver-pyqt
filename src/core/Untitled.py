
import numpy as np
import cv2
import json
import sys
sys.path.append("..")
from core import  FileIO

fio = FileIO.FileIO()
fio.load_project_from_filedir("/home/veily/桌面/LiGan/Pose6dSolver-pyqt/姿态测量")
cam0 = fio.load_camera_pars(0)
cam1 = fio.load_camera_pars(1)

p3d = fio.load_points3d("solve", 0)

p2d0 = np.array([   
    [106,  83],
    [ 69, 189],
    [ 81, 205],
    [ 60, 211],
    [ 46, 194]
], dtype=float)
p2d1 = np.array([    
    [275, 108], 
    [153, 188],
    [164, 202],
    [156, 203],
    [146, 190]
], dtype=float)

res0 = cv2.solvePnP(np.ascontiguousarray(p3d[:,:3]).reshape((-1, 1, 3)), np.ascontiguousarray(p2d0[:, :2]).reshape((-1, 1, 2)), cam0.intrin[:3, :3], np.zeros(5), flags=cv2.SOLVEPNP_EPNP)
res1 = cv2.solvePnP(np.ascontiguousarray(p3d[:,:3]).reshape((-1, 1, 3)), np.ascontiguousarray(p2d1[:, :2]).reshape((-1, 1, 2)), cam1.intrin[:3, :3], np.zeros(5), flags=cv2.SOLVEPNP_EPNP)


print()

