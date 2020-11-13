
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
from widgets import ManualPoseWidget
    
class EmittingStr(QtCore.QObject):
    sig_print = QtCore.pyqtSignal(str) #定义一个发送str的信号
    def write(self, text):
      self.sig_print.emit(str(text))

class FunctionalWidget(QWidget, Ui_FunctionalWidget.Ui_Form):
    sig_btn_run_clicked = pyqtSignal()
    sig_rtvec_changed   = pyqtSignal(str, np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.debug = parent.debug if parent else True
        self.setObjectName("func")

        self.tab_widget_objs.setObjectName("tabWidgetObjs")
        self.tab_widget_objs.removeTab(1)
        self.tab_widget_objs.removeTab(0)
        self.init_sub_tab_widgets(1)

        self.rbtn_model.setChecked(False)

        # 重定向print()
        if self.debug:
            pass
        else:  # release
            sys.stdout = EmittingStr(sig_print=self.slot_qtprint) 
            sys.stderr = EmittingStr(sig_print=self.slot_qtprint)
        loop = QEventLoop()
        QTimer.singleShot(1000, loop.quit)
        loop.exec_() 
        return

    #接收信号str的信号槽
    def slot_qtprint(self, text):  
        QApplication.processEvents()
        cursor = self.text_edit_outprint.textCursor()  
        cursor.movePosition(QtGui.QTextCursor.End)  
        cursor.insertText(text)  
        self.text_edit_outprint.setTextCursor(cursor)  
        self.text_edit_outprint.ensureCursorVisible()   

    def init_sub_tab_widgets(self, n_objs=1):
        self.n_objs = n_objs
        for i_obj in range(n_objs):
            if self.get_sub_tab_widget(i_obj) is not None:
                continue
            name_obj = "obj_{}".format(i_obj + 1)

            sub_tab = QWidget()
            self.tab_widget_objs.addTab(sub_tab, "物体{}".format(i_obj + 1))
            
            sub_maul_widget = ManualPoseWidget.ManualPoseWidget(self)
            sub_maul_widget.setObjectName(name_obj)
            sub_maul_widget.sig_rtvec_changed.connect(self.slot_send_rtvec_msg)

            layout_tab = QHBoxLayout()
            layout_tab.addWidget(sub_maul_widget)
            sub_tab.setLayout(layout_tab)
        #self.show()
        return

    def get_sub_tab_widget(self, obj: int or str) -> ManualPoseWidget.ManualPoseWidget:
        if isinstance(obj, str):
            pass
        elif isinstance(obj, int):
            obj = "obj_{}".format(obj + 1)
        return self.findChild(ManualPoseWidget.ManualPoseWidget, obj)

    def get_theta0(self, name_obj: str):
        rtvec = self.findChild(ManualPoseWidget.ManualPoseWidget, name_obj).get_rtvec()
        return rtvec
        
    def solt_mode_receive(self, mode: str):
        self.mode = mode

        if self.debug:
            print("[DEBUG]:\t<{}>  MODE SET <{}>".format(self.objectName(), mode))
        return

    def slot_send_rtvec_msg(self, name_obj: str, rtvec: np.ndarray):
        self.sig_rtvec_changed.emit(name_obj, rtvec)

        if self.debug:
            print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_rtvec_changed.signal))
        return

    def slot_accept_solve_result(self, name_obj: str, rtvec: np.ndarray):
        sub_tab_widget = self.get_sub_tab_widget(name_obj)
        sub_tab_widget.set_rtvec(rtvec)
        #self.tab_widget_objs.setCurrentIndex(int(name_obj.split("_")[1]) - 1)#
        return

    @pyqtSlot()
    def on_btn_run_clicked(self):
        print("开始解算:")
        self.sig_btn_run_clicked.emit()

        if self.debug:
            print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_btn_run_clicked.signal))
        pass

    @pyqtSlot()
    def on_btn_save_clicked(self):
        print("保存:")
        # for i_obj in range(self.n_objs):
        #     rtvec = self.get_sub_tab_widget(i_obj).get_rtvec()
        #     self.window().fio.save_theta(i_obj, self.window().i_scene, rtvec)
        for i_cam in range(len(self.window().cams)):
            dock_widget = self.window().visualize_area.get_sub_dock_widget(i_cam)
            dock_widget.slot_save_image()
        if self.debug:
            print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_btn_run_clicked.signal))
        pass


if  __name__ == "__main__": 
    import sys

    app = QtWidgets.QApplication(sys.argv)
    widget = FunctionalWidget(None)
    widget.init_sub_tab_widgets(2)
    widget.show()
    sys.exit(app.exec_())
