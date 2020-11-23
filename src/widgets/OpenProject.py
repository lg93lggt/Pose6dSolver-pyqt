
import sys
import os

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui     import *
from PyQt5.QtCore    import *

sys.path.append("..")
from ui import Ui_OpenProjectDialog 

class OpenProjectDialog(QDialog, Ui_OpenProjectDialog.Ui_Dialog):
    sig_accepted = pyqtSignal(str)
    sig_rejected = pyqtSignal()
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setupUi(self)
        self.debug = parent.debug if parent else True
        self.dir_project = ""

    @pyqtSlot()
    def on_toolButton_clicked(self) -> None:
        openfile_name = QFileDialog.getExistingDirectory(self, "打开文件夹")
        self.lineEdit.setPlainText(openfile_name)
        return 

    @pyqtSlot()
    def on_buttonBox_accepted(self) -> None: # 确认
        tmp_dir = self.lineEdit.text()
        if os.path.exists(tmp_dir) and os.path.isdir(tmp_dir):
            pth_ini =  os.path.join(tmp_dir, "project.ini")
            if os.path.exists(pth_ini):
                print("\n打开工程文件夹:\t", tmp_dir)
                self.parentWidget().fio.load_project_from_filedir(tmp_dir)
            else:
                print("\n未找到ini配置文件:\t", tmp_dir)
        else:
            print("\n工程文件夹不存在:\t", tmp_dir)

            if self.debug:
                print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_accepted.signal))

    @pyqtSlot()
    def on_buttonBox_rejected(self): # 取消
        print("\n取消打开工程.")
        self.sig_rejected.emit()

        if self.debug:
            print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_rejected.signal))
        return

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    widget = OpenProjectDialog()
    widget.show()
    sys.exit(app.exec_())
