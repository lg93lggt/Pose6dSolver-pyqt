# -*- coding: utf-8 -*-
# Created by: PyQt5 UI code generator 5.15.1

import os
import sys
import numpy as np
import cv2
from easydict import EasyDict
from  typing import *

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui     import *
from PyQt5.QtCore    import *


sys.path.append("..")
from ui import *
from core import *
from core.SolverPoses6d import SolverPoses6dDLT
from core import InitializerPose6d
from widgets import DockGraphWidget, EditProject, FunctionalWidget, ManualPoseWidget, NewProject, OpenProject, TableWidget, VisualizeWidget, SettingsDialog
from widgets.SettingsDialog import FLAGS_CALIB, FLAGS_THETA0, FLAGS_OPT, FLAGS_POINTS2D 


class MainWindow(QMainWindow, Ui_MainWindow.Ui_MainWindow):
    

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

        return

    # 测量-------------------------------------------------------------------#
    def slot_solve(self):
        print("\n" + "*"*10 + " 测量模式" + "*"*10)
        self.mode = "solve"
        self.fio.match_pairs("solve")
        [self.objs, self.cams] = self.fio.update_mode(self.mode)

        self.slot_init_widgets()
        self.functional_area.sig_rtvec_changed.connect(self.visualize_area.slot_send_new_retvec)
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
            print("\n" + "*"*10 + " 解算相机参数 " + "*"*10)
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


        elif self.mode == "solve":
            print("\n" + "*"*10 + " 解算姿态 " + "*"*10)
            n_cams = self.fio.struct[self.mode].n_cams
            n_objs = self.fio.struct[self.mode].n_objs
            n_scenes = self.fio.struct[self.mode].n_scenes

            is_data_ready = False
            #for i_obj in range(n_objs):
            i_obj = self.functional_area.tab_widget_objs.currentIndex()
            print("物体: {} / {}".format(i_obj + 1, n_objs))
            points3d = self.objs[i_obj].points3d
            points2d_n_cams = []
            points3d_n_cams = []
            cams            = []
            for i_cam in range(n_cams):
                points2d  = self.objs[i_obj].views[i_cam].points2d
                indexes3d = self.objs[i_obj].views[i_cam].indexes3d
                if (points2d is  None) or (indexes3d is None):
                    print("相机{}/{} 未选择, 跳过.".format(i_cam + 1, n_cams))
                    continue
                else:
                    print("相机{}/{} 选择.".format(i_cam + 1, n_cams))
                    cams.append(self.cams[i_cam])
                    points2d_n_cams.append(points2d.astype(float))
                    points3d_n_cams.append(points3d[indexes3d])
                    is_data_ready = True

            if is_data_ready:
                kwargs_data                    = EasyDict({})
                kwargs_data.cameras            = cams
                kwargs_data.points2d_of_n_cams = points2d_n_cams
                kwargs_data.points3d_of_n_cams = points3d_n_cams

                if   self.dialog_settings.settings.FLAGS_THETA0 == FLAGS_THETA0.MAMUAL.value:
                    theta0 = self.functional_area.get_sub_tab_widget(i_obj).get_rtvec()
                elif self.dialog_settings.settings.FLAGS_THETA0 == FLAGS_THETA0.EPNP.value:
                    self.initializer = InitializerPose6d.InitializerPose6d(method="EPnP")
                    theta0 = self.initializer.run_by_epnp(**kwargs_data)
                elif self.dialog_settings.settings.FLAGS_THETA0 == FLAGS_THETA0.PSO.value:
                    kwargs_pso = self.dialog_settings.settings.hyper_params_pso
                    self.initializer = InitializerPose6d.InitializerPose6d(method="PSO")
                    self.initializer.init_pso(**kwargs_pso)
                    theta0 = self.initializer.run_by_pso(**kwargs_data)
                elif self.dialog_settings.settings.FLAGS_THETA0 == FLAGS_THETA0.NONE.value:
                    theta0 = np.zeros(6)
                else:
                    raise ValueError("错误: FLAGS_THETA0 无对应.")
                    return
                    
                if   self.dialog_settings.settings.FLAGS_POINTS2D == FLAGS_POINTS2D.ELLIPSE.value:
                    if i_obj == 0:
                        print("暂无此功能")
                        pass 
                        # TODO 椭圆优化
                elif self.dialog_settings.settings.FLAGS_POINTS2D == FLAGS_POINTS2D.CORRESPOND.value:
                    if   self.dialog_settings.settings.FLAGS_OPT == FLAGS_OPT.ADAM.value:
                        kwargs_adam = self.dialog_settings.settings.hyper_params_adam
                        self.solver = SolverPoses6dDLT(method="Adam", **kwargs_adam)
                    elif self.dialog_settings.settings.FLAGS_OPT == FLAGS_OPT.LM.value:
                        kwargs_lm = self.dialog_settings.settings.lm
                        self.solver = SolverPoses6dDLT(method="LM", **kwargs_lm)
                    else:
                        raise ValueError("错误: FLAGS_OPT 无对应.")
                log   = self.solver.run(theta0, **kwargs_data)
                theta = self.solver.opt.theta
                
                self.fio.save_log(self.mode, self.i_scene, i_obj, log)
                self.fio.save_theta(self.i_scene, i_obj, self.solver.opt.theta)

                self.objs[i_obj].pose = geometry.rtvec_to_rtmat(theta)
                for i_cam in range(n_cams):
                    self.visualize_area.get_sub_dock_widget(i_cam).draw_all()

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