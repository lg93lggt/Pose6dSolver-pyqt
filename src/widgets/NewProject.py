
import sys
import os

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui     import *
from PyQt5.QtCore    import *

sys.path.append("..")
from ui import Ui_NewProjectDialog
from core import FileIO
from widgets import MainWindow  



class NewProjectDialog(QDialog, Ui_NewProjectDialog.Ui_Dialog):
    sig_accepted = pyqtSignal(str)
    sig_rejected = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.debug = parent.debug if parent else True

        self.dir_project = ""
        self.setupUi(self)

    @pyqtSlot()
    def on_toolButton_clicked(self): # 按下
        openfile_name = QFileDialog.getSaveFileName(self, "新建文件夹", "姿态测量", "文件夹")
        self.lineEdit.setPlainText(openfile_name[0])
        return 

    @pyqtSlot()
    def on_buttonBox_accepted(self): # 确认
        tmp_dir = self.lineEdit.text()
        if not os.path.exists(tmp_dir):
            self.dir_project = tmp_dir
            print("\n新建工程文件夹:")
            self.parentWidget().fio.new_project(tmp_dir)
        else:
            print("\t工程文件夹已存在.")

        if self.debug:
            print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_accepted.signal))
        return

    @pyqtSlot()
    def on_buttonBox_rejected(self): # 取消
        print("\n取消新建工程.")

        if self.debug:
            print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_rejected.signal))
        return

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    widget = NewProjectDialog()
    widget.show()
    sys.exit(app.exec_())
