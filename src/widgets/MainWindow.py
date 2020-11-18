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
from easydict import EasyDict


sys.path.append("..")
from ui import *
from core import *
from widgets import DockGraphWidget, EditProject, FunctionalWidget, ManualPoseWidget, NewProject, OpenProject, TableWidget, VisualizeWidget, SettingsDialog
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
        self.scenes_table_area   = TableWidget.ScenesTableWidget(self)
        self.visualize_area      = VisualizeWidget.VisualizeWidget(self)
        self.functional_area     = FunctionalWidget.FunctionalWidget(self)
        self.dialog_new_project  = NewProject.NewProjectDialog(self)
        self.dialog_open_project = OpenProject.OpenProjectDialog(self)
        self.dialog_edit_project = EditProject.EditProjectWidget(self)
        self.dialog_settings     = SettingsDialog.SettingsDialog(self)

        # Lv1控件设置大小
        self.scenes_table_area.setMaximumWidth(200)
        # self.visualize_area.setMinimumWidth(800)
        # self.functional_area.setMaximumWidth(400)

        # Lv1控件设置名称
        self.scenes_table_area.setObjectName("tableScenesArea")
        self.visualize_area.setObjectName("visualizeArea")
        self.functional_area.setObjectName("functionalArea")
        self.dialog_new_project.setObjectName("newProjectDialog")
        self.dialog_open_project.setObjectName("openProjectDialog")
        self.dialog_edit_project.setObjectName("editProgramWidget")

        # Lv2控件初始化n_obj=1
        self.visualize_area.init_sub_dock_widgets(n_cams=1)
        self.functional_area.init_sub_tab_widgets(n_objs=1)

        self.slot_init_widgets()

        # 排版
        self.layout_main.addWidget(self.scenes_table_area, 0, 0)
        self.layout_main.addWidget(self.visualize_area, 0, 1)
        self.layout_main.addWidget(self.functional_area, 0, 5)

        self.action_new_project.triggered.connect(self.slot_new_project)
        self.action_open_project.triggered.connect(self.slot_open_project)
        self.action_edit_project.triggered.connect(self.slot_edit_project)

        self.action_calib.triggered.connect(self.slot_calib)
        self.action_solve.triggered.connect(self.slot_solve)
        self.action_settings.triggered.connect(self.dialog_settings.show)

        self.scenes_table_area.sig_tabel_double_clicked.connect(self.slot_update_scene)

        self.functional_area.btn_run.clicked.connect(self.slot_run_with_mode)

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
    
    def slot_creat_new_project(self, pth_new_project: str) -> None:
        self.fio.new_project(pth_new_project)
        self.dialog_edit_project.init_fio(self.fio)
        return

    # 新建工程-------------------------------------------------------------------#
    
    def slot_new_project(self) -> None:
        print("\n" + "*"*10 + "新建工程" + "*"*10)
        self.dialog_new_project.show()
        return 

    def slot_open_project(self, pth_project: str) -> None:
        print("\n" + "*"*10 + "打开工程" + "*"*10)
        self.dialog_open_project.show()
        return

    # 编辑工程-------------------------------------------------------------------#
    def slot_edit_project(self) -> None:
        print("\n" + "*"*10 + "编辑工程" + "*"*10)
        self.dialog_edit_project.init_fio()
        self.dialog_edit_project.show()
        return 
    
    def slot_refresh_fio(self):
        self.fio.set_unit_length( self.dialog_edit_project.line_unit_length_calib.text())
        self.fio._update()
        return

    # 标定-------------------------------------------------------------------#
    def slot_calib(self):
        print("\n" + "*"*10 + "标定模式" + "*"*10)
        self.mode = "calib"
        self.fio.match_pairs("calib")
        [self.objs, self.cams] = self.fio.update_mode(self.mode)

        self.slot_init_widgets()

        if self.debug:
            print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_mode_calib_activated.signal))
        return

    # 测量-------------------------------------------------------------------#
    def slot_solve(self):
        print("\n" + "*"*10 + " 测量模式" + "*"*10)
        self.mode = "solve"
        self.fio.match_pairs("solve")
        [self.objs, self.cams] = self.fio.update_mode(self.mode)

        self.slot_init_widgets()
        self.functional_area.sig_rtvec_changed.connect(self.visualize_area.slot_send_new_retvec)

        if self.debug:
            print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_mode_solve_activated.signal))
        return

    
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

        self.visualize_area.init_sub_dock_widgets(n_cams)

        for i_cam in range(self.visualize_area.n_cams):
            sub_dock_widget = self.visualize_area.get_sub_dock_widget(i_cam)
            sub_dock_widget.init_sub_table_widgets(n_objs)
            sub_dock_widget.sig_sub_tabel_double_clicked.connect(self.slot_update_obj)
            for i_obj in range(n_objs):
                sub_dock_widget.get_sub_table_view(i_obj).init_array(self.objs[i_obj].points3d)

        self.functional_area.init_sub_tab_widgets(n_objs)
        return

    
    def slot_update_scene(self, i_row: int, i_col: int):
        mode = self.mode
        self.i_scene = i_row
        [n_scenes, n_cams, n_objs] = [self.fio.struct[mode].n_scenes, self.fio.struct[mode].n_cams, self.fio.struct[mode].n_objs]
        print("\n" + "*"*10 + " 场景{} ".format(self.i_scene + 1) + "*"*10)

        print("\n加载图片:")
        for i_cam in range(n_cams):
            ret, image = self.fio.load_image_raw(self.mode, self.i_scene, i_cam)
            if ret:
                self.visualize_area.get_sub_dock_widget(i_cam).init_img(image)

        print("\n加载已选2D点:")
        for i_obj in range(n_objs):
            for i_cam in range(n_cams):
                # 加载2d点
                ret, pts2d = self.fio.load_points2d(self.mode, self.i_scene, i_obj, i_cam)
                if ret:
                    self.objs[i_obj].views[i_cam].points2d = pts2d
        
        print("\n加载已选3D点:")
        for i_obj in range(n_objs):
            for i_cam in range(n_cams):
                # 加载3d点
                ret, indexes = self.fio.load_indexes3d(self.mode, self.i_scene, i_obj, i_cam)
                if ret:
                    self.objs[i_obj].views[i_cam].indexes3d = indexes
                    #self.objs[i_obj].views[i_cam].points3d_chosen = self.objs[i_obj].points3d[self.objs[i_obj].views[i_cam].indexes]
                else:
                    self.objs[i_obj].views[i_cam].indexes3d = None

        print("\n加载姿态:")
        for i_obj in range(n_objs):
            ret, theta = self.fio.load_theta(self.i_scene, i_obj)
            if ret:
                self.objs[i_obj].pose = geometry.rtvec_to_rtmat(theta)
            else:
                self.objs[i_obj].pose = np.eye(4)
        for i_cam in range(n_cams):
            self.visualize_area.get_sub_dock_widget(i_cam)._init_table_widget_show_points()
            self.visualize_area.get_sub_dock_widget(i_cam).draw_all()

        return

    
    def slot_update_obj(self, i_row, i_col):
        print("click", i_row)
        return

    
    def slot_save_points2d(self, name_cam: str, name_obj: str, points2d_n_objs: Dict, points3d_chosen: np.ndarray, indexes_chosen: np.ndarray):
        self.fio.save_points2d(self.mode, self.i_scene, name_obj, name_cam, points2d_n_objs[name_obj])
        self.fio.save_indexes3d(self.mode, self.i_scene, name_obj, name_cam, points3d_chosen, indexes_chosen)
        return

    
    def slot_run_with_mode(self):
        if self.mode == "calib":
            n_cams = self.fio.struct[self.mode].n_cams
            n_objs = self.fio.struct[self.mode].n_objs
            n_scenes = self.fio.struct[self.mode].n_scenes
            for i_cam in range(n_cams):
                print ("相机标定{:d} / {:d}:".format(i_cam + 1, n_cams))
                self.calibrator = CalibratorByDLT.CalibratorByDLT(8, 1)
                points2d  = self.objs[0].views[i_cam].points2d
                indexes3d = self.objs[0].views[i_cam].indexes3d
                points3d  = self.objs[0].points3d[indexes3d]
                self.calibrator.set_points3d(points3d)
                self.calibrator.set_points2d(points2d)
                self.calibrator.run()
                self.fio.save_camera_pars(i_cam, self.calibrator.camera_pars)
                self.visualize_area.get_sub_dock_widget(i_cam).draw_all()
                # else:
                #     print("标定失败.")

                if self.debug:
                    print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_calibrate_successed.signal))

        elif self.mode == "solve":
            n_cams = self.fio.struct[self.mode].n_cams
            n_objs = self.fio.struct[self.mode].n_objs
            n_scenes = self.fio.struct[self.mode].n_scenes

            is_data_ready = False
            for i_obj in range(n_objs):
                print("物体: {} / {}".format(i_obj + 1, n_objs))
                points3d = self.objs[i_obj].points3d
                points2d_n_cams = []
                points3d_n_cams = []
                cams            = []
                for i_cam in range(n_cams):
                    points2d  = self.objs[i_obj].views[i_cam].points2d
                    indexes3d = self.objs[i_obj].views[i_cam].indexes3d
                    if (points2d is  None) or (indexes3d is None):
                        print("物体未选择, 跳过.".format(i_obj + 1))
                        continue
                    else:
                        cams.append(self.cams[i_cam])
                        points2d_n_cams.append(points2d.astype(float))
                        points3d_n_cams.append(points3d[indexes3d])
                        is_data_ready = True

                if is_data_ready:
                    if self.dialog_settings.settings.FLAGS_POINTS2D == 0:
                        if i_obj == 0:
                            self.solver = SolverPoses6d.load_setttings(self.dialog_settings.settings.FLAGS_POINTS2D, self.dialog_settings.settings)
                    else: # TODO
                        n_iters = self.dialog_settings.settings.hyper_params_adam.n_iters
                        alpha   = self.dialog_settings.settings.hyper_params_adam.alpha
                        beta1   = self.dialog_settings.settings.hyper_params_adam.beta1
                        beta2   = self.dialog_settings.settings.hyper_params_adam.beta2
                        self.solver = SolverPoses6d.load_setttings(self.dialog_settings.settings.FLAGS_POINTS2D, self.dialog_settings.settings)
                    self.solver.set_cameras_pars(cams)
                    self.solver.set_points2d_of_n_cams(points2d_n_cams)    
                    self.solver.set_points3d_of_n_cams(points3d_n_cams)
                    # _, r0, t0 = cv2.solvePnP(
                    #     np.ascontiguousarray(points3d_n_cams[0][:,:3]).reshape((-1, 1, 3)), 
                    #     np.ascontiguousarray(points2d_n_cams[0][:, :2]).reshape((-1, 1, 2)), 
                    #     self.cams[0].intrin[:3, :3], 
                    #     np.zeros(5), 
                    #     flags=cv2.SOLVEPNP_EPNP
                    # )     
                    # theta0 = np.hstack((r0.flatten(), t0.flatten()))
                    # np.array([-0.07499097,  0.0564026 , -0.80130748, -0.15415598,  0.15208027, -0.0339067 ])
                    theta0 = self.functional_area.get_sub_tab_widget(i_obj).get_rtvec()
                    log   = self.solver.run(theta0)
                    theta = self.solver.opt.theta
                    
                    #array([ 3.84508410e+02, -2.65103155e+02,  1.91306264e+02, -7.07351854e-02, 5.49107848e-02,  4.32552633e-01])
                    self.fio.save_log(self.mode, self.i_scene, i_obj, log)
                    self.fio.save_theta(self.i_scene, i_obj, self.solver.opt.theta)

                    self.objs[i_obj].pose = geometry.rtvec_to_rtmat(theta)
                    for i_cam in range(n_cams):
                        self.visualize_area.get_sub_dock_widget(i_cam).draw_all()
                    if self.debug:
                        print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_solve_successed.signal))
        else:
            print("错误: 未选择功能.")
        return


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    widget = MainWindow()
    widget.dialog_open_project.plainTextEdit.setPlainText("")
    widget.show()
    sys.exit(app.exec_())