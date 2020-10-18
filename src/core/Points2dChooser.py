
import argparse
import glob
import os
import sys
from typing import *

import cv2
import numpy as np

from . import FileIO


class Points2dChooser():

    def __init__(self, n_points: int=8) -> None:
        self.name_window = "chosse point"
        self.n_points = n_points
        self.points = np.zeros((self.n_points, 2), dtype=np.int)
        self.idx_current_point = 0
        self.is_save_txt = False
        return

    def set_image(self, img: np.ndarray):
        self.img = img.copy()
        return

    def set_dir_output(self, dir_output: str):
        self.dir_output = dir_output
        if not os.path.exists(self.dir_output):
            os.mkdir(self.dir_output)
        return

    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            if self.idx_current_point < self.n_points:
                cv2.circle(self.img, (x, y), 1, (0, 0, 255), 1, 0)
                self.points[self.idx_current_point] = np.array([x, y])
                self.idx_current_point += 1
                print(self.idx_current_point, " : ", x, y)
        return
            
    def run(self):
        print("在图片中选择n个点, 按q键退出, 按o键保存.")
        cv2.namedWindow(self.name_window, cv2.WINDOW_FREERATIO)
        cv2.setMouseCallback(self.name_window, self.mouse_event)
        have_print = True
        while True:
            cv2.imshow(self.name_window, self.img)
            key = cv2.waitKey(10)

            if self.idx_current_point == self.n_points:
                if have_print:
                    print("\r按o键保存", flush=True)
                    have_print = False

                if key == ord("o"):
                    self.is_save_txt = True
                    break

            if key == ord("q"):
                self.is_save_txt = False
                break

class Points2dChooserMultiChannels(Points2dChooser):
    def __init__(self, n_points: int=8, n_cams: int=1) -> None:
        if n_cams == 1:
            super().__init__(n_points)
            return
        else:
            self.n_points = n_points
            self.n_cams = n_cams
            self.names_window = []
            self.is_save_txt_n_cams = []
            self.idx_current_point_n_cams = []
            self.points_n_cams = np.zeros((self.n_cams, self.n_points, 2), dtype=np.int)
            self.name_to_i_cam = {}
            for i_cam in range(self.n_cams):
                name = "cam_" + str(i_cam + 1)
                self.names_window.append(name)
                self.is_save_txt_n_cams.append(False)
                self.idx_current_point_n_cams.append(0)
                self.name_to_i_cam[name] = i_cam
        
            return

    def set_images(self, imgs: List[np.ndarray]):
        self.imgs = imgs.copy()
        return


    def mouse_event(self, event, x, y, flags, param):
        
        if event == cv2.EVENT_LBUTTONDBLCLK:
            name_window = param
            i_cam = self.name_to_i_cam[name_window]
            if self.idx_current_point_n_cams[i_cam] < self.n_points:
                i_point = self.idx_current_point_n_cams[i_cam]
                cv2.circle(self.imgs[i_cam], (x, y), 1, (0, 0, 255), 1, 0)
                self.points_n_cams[i_cam][i_point] = np.array([x, y])
                self.idx_current_point_n_cams[i_cam] += 1
                print("{} points_{}:\t({}, {})".format(name_window, i_point + 1, x, y))
        return
            
    def run(self):
        print("在图片中选择n个点, 按q键退出, 按o键保存.")
        for i_cam in range(self.n_cams):
            name_window = self.names_window[i_cam]
            cv2.namedWindow(name_window, cv2.WINDOW_FREERATIO)
            cv2.setMouseCallback(name_window, self.mouse_event, name_window)
        have_print = [True, True]
        is_quit = False
        key = -1
        while True:
            for i_cam in range(self.n_cams):
                cv2.imshow(self.names_window[i_cam], self.imgs[i_cam])
                key = cv2.waitKey(10)

                if self.idx_current_point_n_cams[i_cam] == self.n_points:
                    if have_print[i_cam]:
                        print("\r按o键保存", flush=True)
                        have_print[i_cam] = False

                    if key == ord("o"):
                        self.is_save_txt_n_cams[i_cam] = True
                    if key == ord("q"):
                        self.is_save_txt_n_cams[i_cam] = False
    
            if np.array(self.is_save_txt_n_cams).all():
                print("所有相机选点完毕.")
                break
            elif (key == ord("q")):
                print("放弃保存文件.")
                break
            else:
                continue

    
def main(args_cmd, **k_args):
    n_points = args_cmd[0]
    mode = k_args["mode"]
    fio = FileIO.FileIO()
    fio  = k_args["fio"]

    if not (mode in fio.file_structure.keys()):
        raise ValueError("mode 设置错误 !")

    n_cams   = fio.file_structure[mode]["n_cams"]
    n_senses = fio.file_structure[mode]["n_senses"]
    dir_images   = fio.file_structure[mode]["dirs"]["images"]
    dir_points2d = fio.file_structure[mode]["dirs"]["points2d"]
    names_subdir = fio.file_structure[mode]["names_subdir"]

    for i_sense in range(n_senses):
        program = Points2dChooserMultiChannels(n_points, n_cams)
        pair = fio.file_structure[mode]["pairs"][i_sense]
        suffix_image = fio.file_structure[mode]["suffix_image"]
        imgs = []
        for i_cam in range(n_cams):
            name_subdir  = names_subdir[i_cam]
            prefix_image = pair[i_cam]
            pth_image = os.path.join(dir_images, name_subdir, prefix_image + suffix_image)
            print("\n", pth_image)
            img = cv2.imread(pth_image)
            imgs.append(img)
        program.set_images(imgs)
        program.run()
        if np.array(program.is_save_txt_n_cams).all():
            for i_cam in range(n_cams):
                prefix = pair[i_cam]
                dir_output = os.path.join(dir_points2d, names_subdir[i_cam])
                fio.save_chosen_points(dir_output, prefix, program.points_n_cams[i_cam])
    return

