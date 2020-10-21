
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
        self.init_sub_tab_widgets(1)

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

    def init_sub_tab_widgets(self, n_obj=1):
        #self.tab_widget_objs.clear()
        for i_obj in range(n_obj):
            if self.get_sub_tab_widget(i_obj) is not None:
                continue
            name_obj = "obj_{}".format(i_obj + 1)

            sub_tab = QWidget()
            self.tab_widget_objs.addTab(sub_tab, "物体{}".format(i_obj + 1))
            
            sub_maul_widget = ManualPoseWidget.ManualPoseWidget(self)
            sub_maul_widget.setObjectName(name_obj)
            sub_maul_widget.sig_rtvec_changed.connect(self.slot_send_rtvec_msg)

            layout_tab = QVBoxLayout()
            sub_tab.setLayout(layout_tab)
            layout_tab.addWidget(sub_maul_widget)
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
        self.tab_widget_objs.setCurrentIndex(name_obj)#int(name_obj.split("_")[1]))
        return

    @pyqtSlot()
    def on_btn_run_clicked(self):
        print("开始解算:")
        self.sig_btn_run_clicked.emit()

        if self.debug:
            print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_btn_run_clicked.signal))
        pass


if  __name__ == "__main__": 
    import sys

    app = QtWidgets.QApplication(sys.argv)
    widget = FunctionalWidget(None)
    widget.show()
    sys.exit(app.exec_())
