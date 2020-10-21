
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
    
class ManualPoseWidget(QWidget, Ui_ManualPoseWidget.Ui_Form):
    sig_rtvec_changed = pyqtSignal(str, np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.debug = parent.debug if parent else True

        # 命名子控件
        self.double_spin_box_rx.setObjectName("lineRx")
        self.double_spin_box_ry.setObjectName("lineRy")
        self.double_spin_box_rz.setObjectName("lineRz")
        self.double_spin_box_tx.setObjectName("lineTx")
        self.double_spin_box_ty.setObjectName("lineTy")
        self.double_spin_box_tz.setObjectName("lineTz")

        self.double_spin_box_rx.setValue(0.)
        self.double_spin_box_ry.setValue(0.)
        self.double_spin_box_rz.setValue(0.)
        self.double_spin_box_tx.setValue(0.)
        self.double_spin_box_ty.setValue(0.)
        self.double_spin_box_tz.setValue(0.)

        # 激活pyqtSlot装饰器
        QtCore.QMetaObject.connectSlotsByName(self)
        return

    def get_rtvec(self):
        try:
            rx = float(self.double_spin_box_rx.value()) if (self.double_spin_box_rx.value() != "") else 0.
        except :
            print("输入必须可转化为数字.")
            rx = 0

        try:
            ry = float(self.double_spin_box_ry.value()) if (self.double_spin_box_ry.value() != "") else 0.
        except :
            print("输入必须可转化为数字.")
            ry = 0

        try:
            rz = float(self.double_spin_box_rz.value()) if (self.double_spin_box_rz.value() != "") else 0.
        except :
            print("输入必须可转化为数字.")
            rz = 0

        try:
            tx = float(self.double_spin_box_tx.value()) if (self.double_spin_box_tx.value() != "") else 0.
        except :
            print("输入必须可转化为数字.")
            tx = 0

        try:
            ty = float(self.double_spin_box_ty.value()) if (self.double_spin_box_ty.value() != "") else 0.
        except :
            print("输入必须可转化为数字.")
            ty = 0

        try:
            tz = float(self.double_spin_box_tz.value()) if (self.double_spin_box_tz.value() != "") else 0.
        except :
            print("输入必须可转化为数字.")
            tz = 0
        print(np.array([rx, ry, rz, tx, ty, tz]))
        return np.array([rx, ry, rz, tx, ty, tz])

    def set_rtvec(self, rtvec: np.ndarray):
        try:
            self.double_spin_box_rx.setText(str(rtvec[0]))
            self.double_spin_box_ry.setText(str(rtvec[1]))
            self.double_spin_box_rz.setText(str(rtvec[2]))
            self.double_spin_box_tx.setText(str(rtvec[3]))
            self.double_spin_box_ty.setText(str(rtvec[4]))
            self.double_spin_box_tz.setText(str(rtvec[5]))
        except :
            print("rtvec 不正确")


    @pyqtSlot()
    def on_lineRx_editingFinished(self):
        rtvec = self.get_rtvec()
        name_obj = self.objectName()
        self.sig_rtvec_changed.emit(name_obj, rtvec)

        if self.debug:
            print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_rtvec_changed.signal))
        pass

    @pyqtSlot()
    def on_lineRy_editingFinished(self):
        rtvec = self.get_rtvec()
        name_obj = self.objectName()
        self.sig_rtvec_changed.emit(name_obj, rtvec)

        if self.debug:
            print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_rtvec_changed.signal))
        pass

    @pyqtSlot()
    def on_lineRz_editingFinished(self):
        rtvec = self.get_rtvec()
        name_obj = self.objectName()
        self.sig_rtvec_changed.emit(name_obj, rtvec)

        if self.debug:
            print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_rtvec_changed.signal))
        pass
    
    def on_lineTx_editingFinished(self):
        rtvec = self.get_rtvec()
        name_obj = self.objectName()
        self.sig_rtvec_changed.emit(name_obj, rtvec)

        if self.debug:
            print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_rtvec_changed.signal))
        pass

    @pyqtSlot()
    def on_lineTy_editingFinished(self):
        rtvec = self.get_rtvec()
        name_obj = self.objectName()
        self.sig_rtvec_changed.emit(name_obj, rtvec)

        if self.debug:
            print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_rtvec_changed.signal))
        pass

    @pyqtSlot()
    def on_lineTz_editingFinished(self):
        rtvec = self.get_rtvec()
        name_obj = self.objectName()
        self.sig_rtvec_changed.emit(name_obj, rtvec)

        if self.debug:
            print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_rtvec_changed.signal))
        pass
        
    
if  __name__ == "__main__": 
    import sys

    app = QtWidgets.QApplication(sys.argv)
    widget = ManualPoseWidget(None)
    widget.show()
    sys.exit(app.exec_())
