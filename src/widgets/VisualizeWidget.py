
import sys
import cv2
import math
from  typing import *

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui     import *
from PyQt5.QtCore    import *
import numpy as np
from core import Visualizer

from widgets import TableWidget


sys.path.append("..")
from ui import * 
from widgets import DockGraphWidget


class VisualizeWidget(QWidget, Ui_VisualizeWidget.Ui_Form):
    sig_mode_calib_activated      = pyqtSignal(str)
    sig_mode_solve_activated      = pyqtSignal(str)
    sig_choose_points2d_successed = pyqtSignal(str, str, dict, np.ndarray, np.ndarray)
    sig_draw_calib_result         = pyqtSignal(int, dict)
    sig_draw_solve_result         = pyqtSignal(int, int, np.ndarray, dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.debug = parent.debug if parent else True
        self.sub_dock_widgets = []
        self.n_cams = 1
        return 

    # def _update(self, fio):
    #     self.n_cams = n_cams


    def init_sub_dock_widgets(self, n_cams: int=1):
        self.n_cams = n_cams
        n_cols = math.ceil(self.n_cams ** 0.5) 
        if n_cols * (n_cols - 1) >= self.n_cams:
            n_rows = n_cols - 1
        else:
            n_rows = n_cols
            
        locs = []
        for i_row in range(n_rows):
            for i_col in range(n_cols):
                i_loc = i_row * n_cols + i_col
                if i_loc < self.n_cams:
                    locs.append([i_row, i_col])
                else:
                    break
                
        for [i_cam, loc] in enumerate(locs):
            name_cam = "cam_{:d}".format(i_cam + 1)
            if self.get_sub_dock_widget(name_cam) is not None:
                continue
            sub_dock_widget = DockGraphWidget.DockGraphWidget(self)
            sub_dock_widget.setObjectName(name_cam)
            sub_dock_widget.sig_choose_points2d_successed.connect(self.solt_send_message_to_main_widget)
            self.sig_draw_calib_result.connect(sub_dock_widget.slot_draw_calib_result)
            self.sig_draw_solve_result.connect(sub_dock_widget.slot_draw_solve_result)
            self.sub_dock_widgets.append(sub_dock_widget)
            self.layout.addWidget(sub_dock_widget, loc[0], loc[1])
            self.sub_dock_widgets[i_cam].groupbox_visualize.setTitle("相机{:d}".format(i_cam + 1))
        return

    def get_sub_dock_widget(self, cam: str or int):
        if   isinstance(cam, int):
            cam = "cam_{}".format(cam + 1) 
            return self.findChild( DockGraphWidget.DockGraphWidget, cam)
        elif isinstance(cam, str):
            return self.findChild( DockGraphWidget.DockGraphWidget, cam)


    def solt_mode_receive(self, mode: str):
        self.mode = mode
        print("{} mode = {}".format(self.objectName(), mode))
        for i_cam in range(self.n_cams):
            sub_dock_widget = self.get_sub_dock_widget(i_cam)
            self.sig_mode_calib_activated.connect(sub_dock_widget.solt_mode_receive)
        self.sig_mode_calib_activated.emit(mode)

        if self.debug:
            print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_mode_calib_activated.signal))
        return

    def solt_send_message_to_main_widget(self, name_cam: str, name_obj: str, points2d: Dict, points3d_chosen: np.ndarray, indexes_chosen):
        self.sig_choose_points2d_successed.emit(name_cam, name_obj, points2d, points3d_chosen, indexes_chosen)

        if self.debug:
            print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_choose_points2d_successed.signal))
        return

    def slot_accept_calibrate_result(self, i_cam: str, camera_pars: Dict):
        self.sig_draw_calib_result.emit(i_cam, camera_pars)

        if self.debug:
            print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_draw_calib_result.signal))
        return

    def slot_accept_solve_result(self, i_obj: str, theta: np.ndarray, cameras_pars: List[Dict]):
        for i_cam in range(self.n_cams):
            
            self.sig_draw_solve_result.emit(i_obj, i_cam, theta, cameras_pars[i_cam])

            if self.debug:
                print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_draw_calib_result.signal))
        return

    def slot_send_new_retvec(self, name_obj: str, theta: np.ndarray):
        for i_cam in range(self.n_cams):
            sub_dock_widget = self.get_sub_dock_widget(i_cam)
            sub_dock_widget.slot_draw_theta0(name_obj, i_cam, theta)




if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    widget = VisualizeWidget()
    widget.show()
    sys.exit(app.exec_())

