
import argparse
import sys
import cv2

import numpy as np
from easydict import EasyDict

sys.path.append("..")
from core import FileIO, SolverPoses6d, Visualizer
from core import geometry as geo

class Object3d(object):
    def __init__(self, name=None) -> None:
        self.pose_self_to_world = np.eye(4)
        self.name = name
        return

    def _update_pose(self, rtvec_self: np.ndarray):
        rvec_self = rtvec_self[:3]
        tvec_self = rtvec_self[3:]
        R = geo.r_to_R(rvec_self)
        T = geo.t_to_T(tvec_self)
        self.pose_self_to_world = self.pose_self_to_world @ R @ T
        return

    def _translate(self, **kwargs: float):
        rtvec_self = np.zeros(6)
        if "x" in kwargs.keys():
            rtvec_self[3] = kwargs["x"]
        if "y" in kwargs.keys():
            rtvec_self[4] = kwargs["y"]
        if "z" in kwargs.keys():
            rtvec_self[5] = kwargs["z"]
        self._update(rtvec_self)
        return

    def _rotate(self, **kwargs: float):
        rtvec_self = np.zeros(6)
        if "x" in kwargs.keys():
            rtvec_self[0] = kwargs["x"]
        if "y" in kwargs.keys():
            rtvec_self[1] = kwargs["y"]
        if "z" in kwargs.keys():
            rtvec_self[2] = kwargs["z"]
        self._update(rtvec_self)
        return
    
    def _set_model(self, model):
        self.model = model
        return
    
    def _set_points3d(self, points3d):
        self.points3d = points3d
        return


class View(object):
    def __init__(self, name=None) -> None:
        self.name = name
        return
    
    def _set_camera(self, camera) -> None:
        self.camera = camera
        return

    def _set_background_image(self, img_background: np.ndarray):
        self.img_background = img_background
        return

class Scene(object):
    def __init__(self, mode: str, n_objs: int=1, n_cams: int=1) -> None:
        self.mode     = mode
        self.n_cams   = n_cams
        self.n_objs   = n_objs
        self.offset_t = 0.01
        self.offset_r = np.pi / 50
        self._init_views()
        self._init_objects()
        return
    
    def _init_views(self):
        self.views = []
        for i_cam in range(self.n_cams):
            self.views.append(View("cam_{}".format(i_cam + 1)))
        return
    
    def _init_objects(self):
        self.objs = []
        for i_obj in range(self.n_objs):
            self.objs.append(Object3d("obj_{}".format(i_obj + 1)))
        return

    def _update(self):
        self.scene.views[i_cam]._set_background_image(self.fio.load_image_raw(self.scene.mode, i_scene, i_cam))
        self.scene.views[i_cam]._set_camera(self.fio.load_camera_pars(i_cam))



class Core(object):
    def __init__(self):
        self.fio    = FileIO.FileIO()
        self.vis    = Visualizer.Visualizer()
        self.solver = SolverPoses6d
        self.mode   = "init"

    def set_mode(self, mode: str):
        self.mode = mode
        return
    
    def _init_scene(self):
        if self.mode != "init":
            self.scene = Scene(self.mode, n_objs=self.fio.struct[self.mode].n_objs, n_cams=self.fio.struct[self.mode].n_cams)
            self.scene._init_views()
        else:
            pass
        return
    
    def _update_scene(self, i_scene: int=0):
        if self.mode == "init":
            return
        for i_cam in range(self.scene.n_cams):
            self.scene.views[i_cam]._set_background_image(self.fio.load_image_raw(self.scene.mode, i_scene, i_cam))
            self.scene.views[i_cam]._set_camera(self.fio.load_camera_pars(i_cam))
        for i_obj in range(self.scene.n_objs):
            self.scene.objs[i_obj]._set_model(self.fio.load_model(self.mode, i_obj))
            self.scene.objs[i_obj]._set_points3d(self.fio.load_points3d(self.mode, i_obj))
        return

    def run(self, args):
        if args.new:
            self.fio.new_project(args.new)
        if args.open:
            self.fio.load_project_from_filedir(args.open)
            self.set_mode("solve")
            self._init_scene()
            self._update_scene(0)
            print()
        while True:
            for view in self.scene.views:
                cv2.namedWindow(view.name, cv2.WINDOW_FREERATIO)
                cv2.imshow(view.name, view.img_background)
                cv2.waitKey(10)
                self.vis.draw(img=view.img_background, points=self.fio.l)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--open")
    parser.add_argument("--new")
    #x = input("qwe:")
    args = parser.parse_args()
    core = Core()
    core.run(args)

    
