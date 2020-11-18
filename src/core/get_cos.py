import cv2
import numpy as np
import sys
import os
import copy
sys.path.append("..")
sys.path.append("./src")
from core import  FileIO
from core import  Visualizer
from core import geometry as geo
from core import SolverPoses6d

fio = FileIO.FileIO()
fio.load_project_from_filedir("./姿态测量4")
ts = np.zeros(11)

def get_angle(vec1, vec2, rtvec1=np.zeros(6), rtvec2=np.zeros(6)):
    pose1 = geo.rtvec_to_rtmat(rtvec1)
    pose2 = geo.rtvec_to_rtmat(rtvec2)

    vec1_ = (pose1 @ vec1.T).T
    vec2_ = (pose2 @ vec2.T).T

    cos_t = np.dot(vec1_, vec2_) / (np.linalg.norm(vec1_) * np.linalg.norm(vec2_))
    t = np.arccos(cos_t) / np.pi * 180 
    print("ANGLE: ", t)
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
def show(theta):
    pose = geo.rtvec_to_rtmat(theta)
    p3d1 = np.array([
        [0, 0, 0, 1],
        [0, 0, 0.3, 1]
    ])
    p3d2 = np.array([
        [0, 0, 0, 1],
        [0, 0.2, 0, 1]
    ])
    
    p3d_ = (pose @ p3d1.T).T

    vec1 = p3d_[1] - p3d_[0]
    vec2 = p3d2[1] - p3d2[0]
    _,img1 = fio.load_image_raw("solve", i_scene, 0)
    _,img2 = fio.load_image_raw("solve", i_scene, 1)
    log = fio.load_log("solve", 0, 0)
    #Visualizer.draw_log(log)
    _, cam1 = fio.load_camera_pars(0)
    _, cam2 = fio.load_camera_pars(1)
    p2d11 = geo.project_points3d_to_2d(np.zeros(6), cam1.intrin @ cam1.extrin, p3d_[:, :3])
    p2d12 = geo.project_points3d_to_2d(np.zeros(6), cam1.intrin @ cam1.extrin, p3d2[:, :3])

    p2d21 = geo.project_points3d_to_2d(np.zeros(6), cam2.intrin @ cam2.extrin, p3d_[:, :3])
    p2d22 = geo.project_points3d_to_2d(np.zeros(6), cam2.intrin @ cam2.extrin, p3d2[:, :3])

    cv2.line(img1, Visualizer.to_plot(p2d11[0]), Visualizer.to_plot(p2d11[1]), (255, 0, 0), 3)
    cv2.line(img1, Visualizer.to_plot(p2d12[0]), Visualizer.to_plot(p2d12[1]), (0, 0, 255), 3)

    cv2.line(img2, Visualizer.to_plot(p2d21[0]), Visualizer.to_plot(p2d21[1]), (255, 0, 0), 3)
    cv2.line(img2, Visualizer.to_plot(p2d22[0]), Visualizer.to_plot(p2d22[1]), (0, 0, 255), 3)

    cv2.imshow("1", img1)
    cv2.imshow("2", img2)
    cv2.waitKey(500)

#     FileIO.imwrite("/home/veily/桌面/Pose6dSolver-pyqt/report/{}/1.jpg".format(i_scene), img1)
#     FileIO.imwrite("/home/veily/桌面/Pose6dSolver-pyqt/report/{}/2.jpg".format(i_scene), img2)

# 原始 图1
# 点 1:    [ 69 230] [ 66 230]
# 点 2:    [185 281]
# 点 3:    [154 274]
# 点 4:    [171 317] 

i_scene = 19
mode = "solve"
fio.load_project_from_filedir("./姿态测量4")

_, p3d = fio.load_points3d(mode, 0)
cams = []
p2ds = []
p3ds = []
for i_cam in range(2):
    _, p2d = fio.load_points2d(mode, i_scene, 0, i_cam)
    _, cam = fio.load_camera_pars(i_cam)
    _, idxs= fio.load_indexes3d(mode, i_scene, 0, i_cam)

    cams.append(cam)
    p2ds.append(p2d.astype(np.float))
    p3ds.append(p3d[idxs])
solver = SolverPoses6d.SolverPoses6dDLT("Adam", n_iter=10000, alpha=0.001, beta1=0.9, beta2=0.999)

du = np.arange(-4, 5)
dv = np.arange(-4, 5)
dudv = np.zeros((du.size * dv.size, 2))
for i in range(du.size):
    u = du[i]
    for j in range(dv.size):
        v = dv[j]
        dudv[i*du.size + j] = np.array([u, v])

p2ds_ = copy.deepcopy(p2ds)
# 点 1:    [ 69 230]
# 点 2:    [185 281]
# 点 3:    [154 274]
# 点 4:    [171 317] 
_, x0 = fio.load_theta(i_scene, 0)

# ------------------------- 单点偏移----------------------------- 
# for i_cam in range(2):
#     print(i_cam, " / 2")
#     for i_pt in range(4):
#         if i_cam==0 and i_pt==0: 
#             continue

#         print(i_pt, " / 4")
#         p2ds_ = copy.deepcopy(p2ds)
#         data = np.zeros((dudv.shape[0], 4))
#         for i_delta in range(dudv.shape[0]):
#             print(i_delta, " / ", dudv.shape[0])
#             p2ds_[i_cam][i_pt] = p2ds[i_cam][i_pt] + dudv[i_delta]
#             print(p2ds_[i_cam])

#             solver.set_cameras_pars(cams)
#             solver.set_points2d_of_n_cams(p2ds_)    
#             solver.set_points3d(p3ds)

#             n_tries = 1
#             while 1:
#                 n_tries += 1
#                 solver.run()
#                 n_tries = 1
#                 if solver.opt.loss < 3:
#                     x0 = solver.opt.theta
#                     v1 = np.array([0, 0, 0.3, 0])
#                     v2 = np.array([0, 0.2, 0, 0])
#                     angle = get_angle(v1, v2, solver.opt.theta)
#                     show(solver.opt.theta)
#                     data[i_delta, 0] = angle
#                     data[i_delta, 1:3] = dudv[i_delta].copy()
#                     data[i_delta, -1] = solver.opt.loss
#                     break
#                 elif n_tries == 10:
#                     x0 = solver.opt.theta
#                     v1 = np.array([0, 0, 0.3, 0])
#                     v2 = np.array([0, 0.2, 0, 0])
#                     angle = get_angle(v1, v2, solver.opt.theta)
#                     show(solver.opt.theta)
#                     data[i_delta, 0] = angle
#                     data[i_delta, 1:3] = dudv[i_delta].copy()
#                     data[i_delta, -1] = solver.opt.loss
#                     break

#             #print()
#         print()
#         np.savetxt("C:/Users/Li/Desktop/Pose6dSolver-pyqt/姿态测量4/test_cov/cam{}_pt{}.txt".format(i_cam, i_pt), data, fmt="%.2f")


# ------------------------- 整体偏移----------------------------- 
for i_cam in range(2):
    print(i_cam, " / 2")
    p2ds_ = copy.deepcopy(p2ds)
    data = np.zeros((dudv.shape[0], 4))
    for i_delta in range(dudv.shape[0]):
        print(i_delta, " / ", dudv.shape[0])

        p2ds_[i_cam]  += dudv[i_delta]

        solver.set_cameras_pars(cams)
        solver.set_points2d_of_n_cams(p2ds_)    
        solver.set_points3d_of_n_cams(p3ds)

        n_tries = 1
        while 1:
            n_tries += 1
            solver.run()
            if solver.opt.loss < 5:
                x0 = solver.opt.theta
                v1 = np.array([0, 0, 0.3, 0])
                v2 = np.array([0, 0.2, 0, 0])
                angle = get_angle(v1, v2, solver.opt.theta)
                show(solver.opt.theta)
                data[i_delta, 0] = angle
                data[i_delta, 1:3] = dudv[i_delta].copy()
                data[i_delta, -1] = solver.opt.loss
                break
            elif n_tries == 3:
                x0 = solver.opt.theta
                v1 = np.array([0, 0, 0.3, 0])
                v2 = np.array([0, 0.2, 0, 0])
                angle = get_angle(v1, v2, solver.opt.theta)
                show(solver.opt.theta)
                data[i_delta, 0] = angle
                data[i_delta, 1:3] = dudv[i_delta].copy()
                data[i_delta, -1] = solver.opt.loss
                break

            print()
        print()
    np.savetxt("./姿态测量4/test_cov/cam{}_ptall.txt".format(i_cam), data, fmt="%.2f")

