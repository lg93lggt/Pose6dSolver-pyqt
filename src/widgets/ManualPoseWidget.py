
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
        self.line_rx.setObjectName("lineRx")
        self.line_ry.setObjectName("lineRy")
        self.line_rz.setObjectName("lineRz")
        self.line_tx.setObjectName("lineTx")
        self.line_ty.setObjectName("lineTy")
        self.line_tz.setObjectName("lineTz")
        self.line_rx.setText("0")
        self.line_ry.setText("0")
        self.line_rz.setText("0")
        self.line_tx.setText("0")
        self.line_ty.setText("0")
        self.line_tz.setText("0")

        # 激活pyqtSlot装饰器
        QtCore.QMetaObject.connectSlotsByName(self)
        return

    def get_rtvec(self):
        try:
            rx = float(self.line_rx.text()) if (self.line_rx.text() != "") else 0.
        except :
            print("输入必须可转化为数字.")
            rx = 0

        try:
            ry = float(self.line_ry.text()) if (self.line_ry.text() != "") else 0.
        except :
            print("输入必须可转化为数字.")
            ry = 0

        try:
            rz = float(self.line_rz.text()) if (self.line_rz.text() != "") else 0.
        except :
            print("输入必须可转化为数字.")
            rz = 0

        try:
            tx = float(self.line_tx.text()) if (self.line_tx.text() != "") else 0.
        except :
            print("输入必须可转化为数字.")
            tx = 0

        try:
            ty = float(self.line_ty.text()) if (self.line_ty.text() != "") else 0.
        except :
            print("输入必须可转化为数字.")
            ty = 0

        try:
            tz = float(self.line_tz.text()) if (self.line_tz.text() != "") else 0.
        except :
            print("输入必须可转化为数字.")
            tz = 0
        print(np.array([rx, ry, rz, tx, ty, tz]))
        return np.array([rx, ry, rz, tx, ty, tz])

    def set_rtvec(self, rtvec: np.ndarray):
        try:
            self.line_rx.setText(str(rtvec[0]))
            self.line_ry.setText(str(rtvec[1]))
            self.line_rz.setText(str(rtvec[2]))
            self.line_tx.setText(str(rtvec[3]))
            self.line_ty.setText(str(rtvec[4]))
            self.line_tz.setText(str(rtvec[5]))
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
