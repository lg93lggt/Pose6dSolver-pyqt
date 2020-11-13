import cv2
import numpy as np
import sys
import os
sys.path.append("..")
sys.path.append("./src")
from core import  FileIO
from core import  Visualizer
from core import geometry as geo

fio = FileIO.FileIO()
fio.load_project_from_filedir("./姿态测量4")
ts = np.zeros(11)

def get_angle(vec1, vec2, rtvec1=np.zeros(6), rtvec2=np.zeros(6)):
    pose1 = geo.rtvec_to_pose(rtvec1)
    pose2 = geo.rtvec_to_pose(rtvec2)

    vec1_ = (pose1 @ vec1.T).T
    vec2_ = (pose2 @ vec2.T).T

    cos_t = np.dot(vec1_, vec2_) / (np.linalg.norm(vec1_) * np.linalg.norm(vec2_))
    t = np.arccos(cos_t) / np.pi * 180 
    print(t)
    return t

# for i_scene in range(11):
#     ret, theta = fio.load_theta(i_scene, 0)

#     pose = geo.rtvec_to_pose(theta)
    p3d1 = np.array([
        [0, 0, 0, 1],
        [0, 0, 0.3, 1]
    ])
    p3d2 = np.array([
        [0, 0, 0, 1],
        [0, 0, 0.2, 1]
    ])
    p3d_ = (pose @ p3d1.T).T

    vec1 = p3d_[1] - p3d_[0]
    vec2 = p3d2[1] - p3d2[0]

#     cos_t = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
#     t = np.arccos(cos_t) / np.pi * 180 
#     ts[i_scene] = t
#     print(t)

#     _,img1 = fio.load_image_raw("solve", i_scene, 0)
#     _,img2 = fio.load_image_raw("solve", i_scene, 1)
#     log = fio.load_log("solve", 0, 0)
#     #Visualizer.draw_log(log)
#     _, cam1 = fio.load_camera_pars(0)
#     _, cam2 = fio.load_camera_pars(1)
#     p2d11 = geo.project_points3d_to_2d(np.zeros(6), cam1.intrin @ cam1.extrin, p3d_[:, :3])
#     p2d12 = geo.project_points3d_to_2d(np.zeros(6), cam1.intrin @ cam1.extrin, p3d2[:, :3])

#     p2d21 = geo.project_points3d_to_2d(np.zeros(6), cam2.intrin @ cam2.extrin, p3d_[:, :3])
#     p2d22 = geo.project_points3d_to_2d(np.zeros(6), cam2.intrin @ cam2.extrin, p3d2[:, :3])

#     cv2.line(img1, Visualizer.to_plot(p2d11[0]), Visualizer.to_plot(p2d11[1]), (255, 0, 0), 3)
#     cv2.line(img1, Visualizer.to_plot(p2d12[0]), Visualizer.to_plot(p2d12[1]), (0, 0, 255), 3)

#     cv2.line(img2, Visualizer.to_plot(p2d21[0]), Visualizer.to_plot(p2d21[1]), (255, 0, 0), 3)
#     cv2.line(img2, Visualizer.to_plot(p2d22[0]), Visualizer.to_plot(p2d22[1]), (0, 0, 255), 3)

#     cv2.imshow("1", img1)
#     cv2.imshow("2", img2)
#     cv2.waitKey(1)

#     FileIO.imwrite("/home/veily/桌面/Pose6dSolver-pyqt/report/{}/1.jpg".format(i_scene), img1)
#     FileIO.imwrite("/home/veily/桌面/Pose6dSolver-pyqt/report/{}/2.jpg".format(i_scene), img2)

# 原始 图1
# 点 1:    [ 69 230] [ 66 230]
# 点 2:    [185 281]
# 点 3:    [154 274]
# 点 4:    [171 317] 

i_scene = 19
log = fio.load_log("solve", i_scene, 0)
res = log[np.argmin(log[:, -1])]
theta = res[:6]


v1 = np.array([0, 0, 0.3, 0])
v2 = np.array([0, 0.2, 0, 0])
angle = get_angle(v1, v2, theta)

id = "u_-1_v_2"
pth = "./姿态测量4/test_cov/{}.txt".format(id)
#np.savetxt(pth, np.hstack((res, angle)))

np.cov()

print()
