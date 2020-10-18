
import sys
import os

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui     import *
from PyQt5.QtCore    import *

sys.path.append("..")
from ui import Ui_EditProject
from core import FileIO


class EditProjectWidget(QDialog, Ui_EditProject.Ui_Form):
    sig_widget_closed              = pyqtSignal()
    sig_unit_length_changed        = pyqtSignal(float)
    sig_choose_images_calib_successed   = pyqtSignal(str)
    sig_choose_images_solve_successed   = pyqtSignal(str)
    sig_choose_points3d_solve_successed = pyqtSignal(str)
    sig_choose_models_solve_successed   = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setupUi(self)
        self.debug = parent.debug if parent else True

        self.pushbtn_add_images_calib.setObjectName("pushbtnAddImagesCalib")
        self.pushbtn_add_images_solve.setObjectName("pushbtnAddImagesSolve")
        self.pushbtn_add_points3d_solve.setObjectName("pushbtnAddPoints3dSolve")
        self.pushbtn_add_models_solve.setObjectName("pushbtnAddModelsSolve")
        self.dir_input_images_calib   = ""
        self.dir_input_images_solve   = ""
        self.dir_input_models_solve   = ""
        self.dir_input_points3d_solve = ""
        QtCore.QMetaObject.connectSlotsByName(self)
        return

    def init_fio(self, fio):
        self.fio = fio
        self._update(fio)
        return

    def _update(self, fio):
        self.line_dir_root.setText(fio.struct.dir_root)
        self.line_dir_images_calib.setText( "images_claib")
        self.line_dir_points2d_calib.setText( "points2d_claib")
        self.line_dir_points3d_calib.setText( "points3d_claib")
        self.line_dir_results_calib.setText( "results_claib")
        self.line_dir_visualize_calib.setText( "visualize_claib")

        self.line_dir_images_solve.setText("images_solve")
        self.line_dir_points2d_solve.setText("points2d_solve")
        self.line_dir_points3d_solve.setText("points3d_solve")
        self.line_dir_models_solve.setText("models_solve")
        self.line_dir_results_solve.setText("results_solve")
        self.line_dir_visualize_solve.setText("visualize_solve")
        self.line_dir_logs_solve.setText("logs_solve")

        if int(fio.struct.calib.n_cams) > 0:
            self.line_num_cameras_calib.setText(str(fio.struct.calib.n_cams))
        else:
            self.line_num_cameras_calib.setText("0")

        if int(fio.struct.calib.n_scenes) > 0:
            self.line_num_scenes_calib.setText(str(fio.struct.calib.n_scenes))
        else:
            self.line_num_scenes_calib.setText("0")

        if float(fio.struct.calib.unit_length) > 0:
            self.line_unit_length_calib.setText(str(fio.struct.calib.unit_length))
        else:
            self.line_unit_length_calib.setText("1")

        if int(fio.struct.solve.n_cams) > 0:
            print(self.line_num_cameras_solve.objectName())
            self.line_num_cameras_solve.setText(str(fio.struct.solve.n_cams))
        else:
            self.line_num_cameras_solve.setText("0")

        if int(fio.struct.solve.n_scenes) > 0:
            self.line_num_scenes_solve.setText(str(fio.struct.solve.n_scenes))
        else:
            self.line_num_scenes_solve.setText("0")

        if int(fio.struct.solve.n_objs) > 0:
            self.line_num_points3d_solve.setText(str(fio.struct.solve.n_objs))
        else:
            self.line_num_scenes_solve.setText("0")

        if int(fio.struct.solve.n_models) > 0:
            self.line_num_models_solve.setText(str(fio.struct.solve.n_models))
        else:
            self.line_num_models_solve.setText("0")
        return
    
    def closeEvent(self, evt: QCloseEvent):
        self.sig_widget_closed.emit()

        if self.debug:
            print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_widget_closed.signal))
        return

    @pyqtSlot()
    def on_pushbtnAddImagesCalib_released(self):
        dir_input_images_calib = QFileDialog.getExistingDirectory(self, "打开标定图像文件夹")
        self.dir_input_images_calib = dir_input_images_calib
        if dir_input_images_calib != "":
            self.sig_choose_images_calib_successed.emit(dir_input_images_calib)

            if self.debug:
                print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_choose_images_calib_successed.signal))
        return 

    @pyqtSlot()
    def on_pushbtnAddImagesSolve_clicked(self):
        dir_input_images_solve = QFileDialog.getExistingDirectory(self, "打开测量图像文件夹")
        self.dir_input_images_solve = dir_input_images_solve
        if dir_input_images_solve != "":
            self.sig_choose_images_solve_successed.emit(dir_input_images_solve)
            print(dir_input_images_solve)

            if self.debug:
                print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_widget_closed.signal))

    @pyqtSlot()
    def on_pushbtnAddPoints3dSolve_clicked(self):
        dir_input_points3d_solve = QFileDialog.getExistingDirectory(self, "打开关键点文件夹")
        self.dir_input_points3d_solve = dir_input_points3d_solve
        if dir_input_points3d_solve != "":
            self.sig_choose_points3d_solve_successed.emit(dir_input_points3d_solve)
            print(dir_input_points3d_solve)

            if self.debug:
                print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_widget_closed.signal))

    @pyqtSlot()
    def on_pushbtnAddModelsSolve_clicked(self):
        dir_input_models_solve = QFileDialog.getExistingDirectory(self, "打开模型文件夹")
        self.dir_input_models_solve = dir_input_models_solve
        if dir_input_models_solve != "":
            self.sig_choose_models_solve_successed.emit(dir_input_models_solve)
            print(dir_input_models_solve)

            if self.debug:
                print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_widget_closed.signal))
        
    @pyqtSlot(float)
    def on_line_unit_length_calib_editingFinished(self):
        unit_length = float(self.line_unit_length_calib.text())
        try:
            unit_length = float(self.line_unit_length_calib.text())
            self.sig_unit_length_changed(unit_length)
            print("设置标定架单位长度:\t".format(unit_length))
        except :
            print("类型错误, 应为数字!")
        return


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    fio = FileIO.FileIO()
    fio.load_project_from_filedir("C:/Users/Li/Desktop/Pose6dSolver-pyqt/姿态测量")
    widget = EditProjectWidget()
    widget.init_fio(fio)
    widget.show()
    sys.exit(app.exec_())
