# -*- coding: utf-8 -*-
# Created by: PyQt5 UI code generator 5.15.1

import os
import sys
import cv2
from  typing import *

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui     import *
from PyQt5.QtCore    import *
import numpy as np


sys.path.append("..")
from ui import *
from core import *
from widgets import *

from slot import * 


class MainWindow(QMainWindow, Ui_MainWindow.Ui_MainWindow):
    sig_mode_calib_activated = pyqtSignal(str)
    sig_mode_solve_activated = pyqtSignal(str)
    sig_calibrate_successed  = pyqtSignal(int, dict)
    sig_solve_successed      = pyqtSignal(int, np.ndarray, list)

    def debug(function):
        print("[DEBUG]: run func: {}()".format(function.__name__))
        #     if kwargs == {}:
        #         return func(*args)
        #     else:
        #         return func(*args, **kwargs)
        # return wrapper  
        return function
    
    def __init__(self, parent=None, debug=True):
        super().__init__(parent)
        self.debug = debug
        self.mode = "init"
        self.i_scene = -1
        self.setupUi(self)
        self.setObjectName("Main")
        self.fio = FileIO.FileIO()

        # Lv1控件初始化
        self.scenes_table_area = TableWidget.ScenesTableWidget(self)
        self.visualize_area    = VisualizeWidget.VisualizeWidget(self)
        self.functional_area   = FunctionalWidget.FunctionalWidget(self)
        self.dialog_new_project  = NewProject.NewProjectDialog(self)
        self.dialog_open_project = OpenProject.OpenProjectDialog(self)
        self.widget_edit_project = EditProject.EditProjectWidget(self)

        # Lv1控件设置名称
        self.scenes_table_area.setObjectName("tableScenesArea")
        self.visualize_area.setObjectName("visualizeArea")
        self.functional_area.setObjectName("functionalArea")
        self.dialog_new_project.setObjectName("newProjectDialog")
        self.dialog_open_project.setObjectName("openProjectDialog")
        self.widget_edit_project.setObjectName("editProgramWidget")

        # Lv2控件初始化n_obj=1
        self.visualize_area.init_sub_dock_widgets(n_cams=1)
        self.functional_area.init_sub_tab_widgets(n_obj=1)

        self.slot_init_widgets()

        self.visualize_area.sig_choose_points2d_successed.connect(self.solt_save_points2d)
        self.sig_calibrate_successed.connect(self.visualize_area.slot_accept_calibrate_result)

        self.functional_area.sig_btn_run_clicked.connect(self.slot_run_with_mode)

        # 排版
        self.layout_main.addWidget(self.scenes_table_area)
        self.layout_main.addWidget(self.visualize_area)
        self.layout_main.addWidget(self.functional_area)

        # 激活pyqtSlot装饰器
        QtCore.QMetaObject.connectSlotsByName(self)
        
        self.setWindowState(Qt.WindowMaximized)  # 界面最大化
        return

    ## 槽函数-------------------------------------------------------------------##
    """
    格式:
        @pyqtSlot(type(*pars))
        def on_<objectname>_<singal>(*pars):
            pass

        <name_sig>.connet(<name_slot>)
    """
    # 新建工程-------------------------------------------------------------------#
    @debug
    @pyqtSlot()
    def on_actionNewProject_triggered(self) -> None:
        self.dialog_new_project.sig_accepted.connect(self.slot_creat_new_project)
        # self.dialog_new_project.sig_rejected.connect() # 取消事件, 暂时不用
        self.dialog_new_project.show()
        return 
    
    @debug
    def slot_creat_new_project(self, pth_new_project: str) -> None:
        self.fio.new_project(pth_new_project)
        self.widget_edit_project.init_fio(self.fio)
        return

    # 打开工程-------------------------------------------------------------------#
    @debug
    @pyqtSlot()
    def on_actionOpenProject_triggered(self) -> None:
        self.dialog_open_project.sig_accepted.connect(self.slot_open_project)
        # self.dialog_new_project.sig_rejected.connect() # 取消事件, 暂时不用
        self.dialog_open_project.show()
        return 

    def slot_open_project(self, pth_project: str) -> None:
        self.fio.load_project_from_filedir(pth_project)
        self.fio._update()
        self.widget_edit_project.init_fio(self.fio)
        return

    # 编辑工程-------------------------------------------------------------------#
    @debug
    @pyqtSlot()
    def on_actionEditProject_triggered(self) -> None:
        self.widget_edit_project.sig_widget_closed.connect(self.slot_refresh_fio)
        self.widget_edit_project.sig_unit_length_changed.connect(self.slot_set_unit_length)
        self.widget_edit_project.sig_choose_images_calib_successed.connect(self.slot_load_images_calib)
        self.widget_edit_project.sig_choose_images_solve_successed.connect(self.slot_load_images_solve)
        self.widget_edit_project.sig_choose_models_solve_successed.connect(self.slot_load_models_solve)
        self.widget_edit_project.sig_choose_points3d_solve_successed.connect(self.slot_load_points3d_solve)
        # self.widget_edit_project.sig_rejected.connect()
        self.widget_edit_project.show()
        return 
    
    @debug
    def slot_refresh_fio(self):
        self.fio.set_unit_length( self.widget_edit_project.line_unit_length_calib.text())
        self.fio._update()
        return
    
    @debug
    def slot_set_unit_length(self, unit_length: float) -> None:
        self.fio.set_unit_length(unit_length)
        return

    @debug
    def slot_load_images_calib(self, dir_images_calib) -> None:
        self.fio.load_images_from_motherfolder_dir(dir_images_calib, "calib")
        self.widget_edit_project._update(self.fio)
        return

    @debug
    def slot_load_images_solve(self, dir_images_solve) -> None:
        self.fio.load_images_from_motherfolder_dir(dir_images_solve, "solve")
        self.widget_edit_project._update(self.fio)
        return

    @debug
    def slot_load_points3d_solve(self, dir_points3d_solve) -> None:
        self.fio.load_points3d_from_motherfolder_dir(dir_points3d_solve, "solve")
        self.widget_edit_project._update(self.fio)
        return

    @debug
    def slot_load_models_solve(self, dir_models_solve) -> None:
        self.fio.load_modeles_from_motherfolder_dir(dir_models_solve, "solve")
        self.widget_edit_project._update(self.fio)
        return

    # 标定-------------------------------------------------------------------#
    @debug
    @pyqtSlot()
    def on_actionCalib_triggered(self):
        print("\n标定模式:")
        self.mode = "calib"
        self.fio.match_pairs("calib")
        self.sig_mode_calib_activated.connect(self.scenes_table_area.solt_mode_receive)
        self.sig_mode_calib_activated.connect(self.visualize_area.solt_mode_receive)
        self.sig_mode_calib_activated.connect(self.functional_area.solt_mode_receive)
        self.sig_mode_calib_activated.connect(self.slot_init_widgets)
        self.sig_mode_calib_activated.emit("calib")

        if self.debug:
            print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_mode_calib_activated.signal))
        return

    # 测量-------------------------------------------------------------------#
    @debug
    @pyqtSlot()
    def on_actionSolve_triggered(self):
        self.mode = "solve"
        self.fio.match_pairs("solve")
        self.sig_mode_solve_activated.connect(self.scenes_table_area.solt_mode_receive)
        self.sig_mode_solve_activated.connect(self.visualize_area.solt_mode_receive)
        self.sig_mode_solve_activated.connect(self.slot_init_widgets)
        self.functional_area.sig_rtvec_changed.connect(self.visualize_area.slot_send_new_retvec)
        self.sig_mode_solve_activated.emit("solve")

        if self.debug:
            print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_mode_solve_activated.signal))
        return

    @debug
    def slot_init_widgets(self):
        """
            初始化子控件,孙控件
        """
        mode = self.mode

        if self.mode == "init":
            [n_scenes, n_cams, n_objs] = [1, 1, 1]
            name_cam = "cam_{:d}".format(1)
            sub_dock_widget = self.visualize_area.findChild(DockGraphWidget.DockGraphWidget, name_cam)
            sub_dock_widget.init_sub_table_widgets(n_objs=1)
            return
            
        [n_scenes, n_cams, n_objs] = [self.fio.struct[mode].n_scenes, self.fio.struct[mode].n_cams, self.fio.struct[mode].n_objs]
        
        # updatae table for all scenes
        self.scenes_table_area.set_shape(n_scenes, n_cams + 2)
        labels_cols = [str(i_scene + 1) for i_scene in range(n_scenes)]
        labels_rows = ["相机{:d}".format(i_cam + 1) for i_cam in range(n_cams)]
        labels_rows.append("标定")
        labels_rows.append("解算")
        self.scenes_table_area.set_shape(n_scenes, n_cams + 1)
        self.scenes_table_area.set_texts(labels_cols, labels_rows)
        self.scenes_table_area.set_checkboxes(i_col=n_cams)
        for i_scene in range(n_scenes):
            data_row = self.fio.struct[mode].images[i_scene]
            self.scenes_table_area.set_datas_row(data_row, i_scene)
 
        # updatae visualize_area
        self.visualize_area.init_sub_dock_widgets(n_cams)
        for i_cam in range(self.visualize_area.n_cams):
            name_cam = "cam_{:d}".format(i_cam + 1)
            sub_dock_widget = self.visualize_area.findChild(DockGraphWidget.DockGraphWidget, name_cam)
            sub_dock_widget.init_sub_table_widgets(n_objs)
            if self.fio.load_camera_pars(i_cam):
                sub_dock_widget.camera_pars = self.fio.load_camera_pars(i_cam)

        self.scenes_table_area.sig_tabel_double_clicked.connect(self.slot_update_scene)
        for i_cam in range(self.visualize_area.n_cams):
            self.visualize_area.get_sub_dock_widget(i_cam).sig_sub_tabel_double_clicked.connect(self.slot_update_obj)

        self.functional_area.init_sub_tab_widgets(n_objs)
        return

    @debug
    def slot_update_scene(self, name_object: str, i_row: int, i_col: int):
        mode = self.mode
        self.i_scene = i_row
        [n_scenes, n_cams, n_objs] = [self.fio.struct[mode].n_scenes, self.fio.struct[mode].n_cams, self.fio.struct[mode].n_objs]
        for i_cam in range(n_cams):
            name_cam = "cam_{:d}".format(i_cam + 1)
            img = self.fio.load_image_raw(mode, self.i_scene, i_cam)
            sub_dock_widget = self.visualize_area.get_sub_dock_widget(name_cam)
            sub_dock_widget.init_img(img)
            points2d_objs = {}
            for i_obj in range(n_objs):
                name_obj = "obj_{:d}".format(i_obj + 1)
                points3d = self.fio.load_points3d(self.mode, i_obj)
                sub_table_widget = sub_dock_widget.findChild(TableWidget.ObjectTableWidget, name_obj)
                sub_table_widget.init_array(points3d)
                print(sub_table_widget.objectName(), "加载模型3d点:", points3d.shape)
                if self.fio.loadz_points3d(self.mode, self.i_scene, i_obj, i_cam) is not None:
                    points2d = self.fio.load_points2d(self.mode, self.i_scene, i_obj, i_cam)
                    if points2d_objs is not None:
                        points2d_objs[name_obj] = points2d
                    else:
                        continue
                    
                    points3d_array_and_indexes = self.fio.loadz_points3d(self.mode, self.i_scene, i_obj, i_cam)
                    points3d_chosen = points3d_array_and_indexes["array"]
                    indexes_chosen  = points3d_array_and_indexes["indexes"]
                    if points3d_chosen.shape[0] > points3d.shape[0]:
                        print("无法匹配模型与已选点:")
                        pth_2d = os.path.join(self.fio.struct.dir_root, "points2d_"+mode, name_cam, name_obj, self.fio.index2name("scene", self.i_scene)+".txt")
                        pth_3d = os.path.join(self.fio.struct.dir_root, "points3d_"+mode, name_cam, name_obj, self.fio.index2name("scene", self.i_scene)+".npz")
                        if os.path.exists(pth_2d):
                            os.remove(pth_2d)
                        if os.path.exists(pth_3d):
                            points3d_array_and_indexes.close()
                            os.remove(pth_3d)
                        print("删除:\n{}\n{}".format(pth_2d, pth_3d))
                    else:
                        sub_table_widget.array_chosen   = points3d_chosen
                        sub_table_widget.indexes_chosen = indexes_chosen
                        print(sub_table_widget.objectName(), "加载已选3d点:", points3d_chosen.shape)
                        print(sub_table_widget.objectName(), "加载已选3d点索引:", indexes_chosen)
            sub_dock_widget.points2d_objs = points2d_objs
            sub_dock_widget._update_table_widget_show_points()
        #self.findChild(DockGraphWidget.DockGraphWidget, "cam_2").findChild(TableWidget.ObjectTableWidget, "obj_1").array
        return

    @debug
    def slot_update_obj(self, i_row, i_col):
        print("click", i_row)
        return

    @debug
    def solt_save_points2d(self, name_cam: str, name_obj: str, points2d_n_objs: Dict, points3d_chosen: np.ndarray, indexes_chosen: np.ndarray):
        self.fio.save_points2d(self.mode, self.i_scene, name_obj, name_cam, points2d_n_objs[name_obj])
        self.fio.savez_points3d(self.mode, self.i_scene, name_obj, name_cam, points3d_chosen, indexes_chosen)
        return

    @debug
    def slot_run_with_mode(self):
        if self.mode == "calib":
            n_cams = self.fio.struct[self.mode].n_cams
            n_objs = self.fio.struct[self.mode].n_objs
            n_scenes = self.fio.struct[self.mode].n_scenes
            for i_cam in range(n_cams):
                print ("相机标定{:d} / {:d}:".format(i_cam + 1, n_cams))
                self.calibrator = CalibratorByDLT.CalibratorByDLT(8, 1)
                self.calibrator.set_points3d(self.fio.loadz_points3d(self.mode, self.i_scene, "obj_1", self.fio.index2name("cam", i_cam))["array"])
                self.calibrator.set_points2d(self.fio.load_points2d(self.mode, self.i_scene, "obj_1", self.fio.index2name("cam", i_cam)))
                self.calibrator.run()
                self.fio.save_camera_pars(i_cam, self.calibrator.camera_pars)
                self.sig_calibrate_successed.emit(i_cam, self.calibrator.camera_pars)

                if self.debug:
                    print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_calibrate_successed.signal))

        elif self.mode == "solve":
            n_cams = self.fio.struct[self.mode].n_cams
            n_objs = self.fio.struct[self.mode].n_objs
            n_scenes = self.fio.struct[self.mode].n_scenes
            cameras_pars = []
            for i_cam in range(n_cams):
                camera_pars = self.fio.load_camera_pars(i_cam)
                if camera_pars is None:
                    return
                else:
                    cameras_pars.append(camera_pars)

            self.models = []
            for i_obj in range(n_objs):
                if self.fio.load_model(self.mode, i_obj):
                    self.models.append(self.fio.load_model(self.mode, i_obj))

            is_data_ready = False
            for i_obj in range(n_objs):
                print("物体: {} / {}".format(i_obj + 1, n_objs))
                points2d_n_cams = []
                points3d_n_cams = []
                for i_cam in range(n_cams):
                    points2d = self.fio.load_points2d(self.mode, self.i_scene, i_obj, i_cam)
                    points3d_npz = self.fio.loadz_points3d(self.mode, self.i_scene, i_obj, i_cam)
                    if (points2d is None) or (points3d_npz is None):
                        print("物体未选择, 跳过.".format(i_obj + 1))
                        is_data_ready = False
                        break
                    points2d_n_cams.append(points2d)
                    points3d_n_cams.append(points3d_npz["array"])
                    is_data_ready = True
                if is_data_ready:

                    pso = ParticleSwarmOptimization.ParticleSwarmOptimization(50, 6, 1000)
  
                    #self.solver = SolverPoses6d.SolverPoses6d("LM", n_iters=1000, alpha=0.01) 
                    self.solver = SolverPoses6d.SolverPoses6d("Adam", n_iters=10000, alpha=0.001, beta1=0.9, beta2=0.99) 
                    self.solver.set_cameras_pars(cameras_pars)
                    self.solver.set_points2d_of_n_cams(points2d_n_cams)    
                    self.solver.set_points3d(points3d_n_cams[0])
                    theta0 = self.functional_area.get_theta0(self.fio.index2name("obj", i_obj))
                    theta0 = geometry.rtvec_degree2rad(theta0)
                    print("theta0:", theta0)
                    log   = self.solver.run(theta0)
                    theta = self.solver.opt.theta
                    
                    theta = geometry.rtvec_rad2degree(theta)

                    self.fio.save_log(self.mode, self.i_scene, i_obj, log)
                    self.fio.save_theta(self.mode, self.i_scene, i_obj, theta)
                    self.sig_solve_successed.connect(self.visualize_area.slot_accept_solve_result)
                    self.sig_solve_successed.connect(self.functional_area.slot_accept_solve_result)
                    self.sig_solve_successed.emit(i_obj, theta, cameras_pars)

                    if self.debug:
                        print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_solve_successed.signal))
                


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    widget = MainWindow()
    widget.dialog_open_project.plainTextEdit.setPlainText("")
    widget.show()
    sys.exit(app.exec_())