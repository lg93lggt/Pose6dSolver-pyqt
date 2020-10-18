
import sys
import os

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui     import *
from PyQt5.QtCore    import *

sys.path.append("..")
from ui import Ui_NewProjectDialog
from core import FileIO



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
        self.plainTextEdit.setPlainText(openfile_name[0])
        return 

    @pyqtSlot()
    def on_buttonBox_accepted(self): # 确认
        tmp_dir = self.plainTextEdit.toPlainText()
        if not os.path.exists(tmp_dir):
            self.dir_project = tmp_dir
            print("\n新建工程文件夹:")
            self.sig_accepted.emit(self.dir_project)

            if self.debug:
                print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_accepted.signal))
            return

    @pyqtSlot()
    def on_buttonBox_rejected(self): # 取消
        print("\n取消.")
        self.sig_rejected.emit()

        if self.debug:
            print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_rejected.signal))
        return

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    widget = NewProjectDialog()
    widget.show()
    sys.exit(app.exec_())
