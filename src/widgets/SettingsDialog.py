
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui     import *
from PyQt5.QtCore    import *
from easydict import EasyDict
import os
import sys
import json
import enum

sys.path.append("..")
from ui import Ui_SettingsDialog

class FLAGS_CALIB(enum.Enum):
    DLT = 0

class FLAGS_THETA0(enum.Enum):
    MAMUAL = 0
    PSO    = 1
    EPNP   = 2
    NONE   = 3

class FLAGS_OPT(enum.Enum):
    ADAM = 0
    LM   = 1

class FLAGS_POINTS2D(enum.Enum):
    ELLIPSE    = 0
    CORRESPOND = 1
    
class SettingsDialog(QDialog, Ui_SettingsDialog.Ui_Dialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        try:
            self.debug = self.window().debug 
        except :
            self.debug = False

        self.settings = EasyDict({})
        self.accepted.connect(self.save_settings)
        # 激活pyqtSlot装饰器
        QtCore.QMetaObject.connectSlotsByName(self)
        self.load_settings()
        return

    def load_settings(self, pth="./settings.ini"):
        print("打开设置文件:", pth)
        
        with open(os.path.abspath(pth)) as f:
            self.settings = EasyDict(json.load(f))

        if self.settings.FLAGS_THETA0 == FLAGS_THETA0.MAMUAL.value:
            self.rbtn_manual.setChecked(True)
        if self.settings.FLAGS_THETA0 == FLAGS_THETA0.PSO.value:
            self.rbtn_pso.setChecked(True)
        if self.settings.FLAGS_THETA0 == FLAGS_THETA0.EPNP.value:
            self.rbtn_epnp.setChecked(True)
        if self.settings.FLAGS_THETA0 == FLAGS_THETA0.NONE.value:
            self.rbtn_none.setChecked(True) 
        

        if self.settings.FLAGS_OPT == FLAGS_OPT.ADAM.value:
            self.rbtn_adam.setChecked(True)
        if self.settings.FLAGS_OPT == FLAGS_OPT.LM.value:
            self.rbtn_lm.setChecked(True)
        
        if self.settings.FLAGS_POINTS2D == FLAGS_POINTS2D.CORRESPOND.value:
            self.rbtn_correspond.setChecked(True)
        if self.settings.FLAGS_POINTS2D == FLAGS_POINTS2D.ELLIPSE.value:
            self.rbtn_ellipse.setChecked(True)

        self.spinbox_niters_adam.setValue(self.settings.hyper_params_adam.n_iters) 
        self.spinbox_alpha_adam.setValue(self.settings.hyper_params_adam.alpha) 
        self.spinbox_beta1_adam.setValue(self.settings.hyper_params_adam.beta1) 
        self.spinbox_beta2_adam.setValue(self.settings.hyper_params_adam.beta2) 

        self.spinbox_niters_lm.setValue(self.settings.hyper_params_lm.n_iters) 
        self.spinbox_alpha_lm.setValue(self.settings.hyper_params_lm.alpha) 

        self.spinbox_niters_pso.setValue(self.settings.hyper_params_pso.n_iters) 
        self.spinbox_npops_pso.setValue(self.settings.hyper_params_pso.n_pops) 
        self.spinbox_w_pso.setValue(self.settings.hyper_params_pso.w) 
        self.spinbox_c1_pso.setValue(self.settings.hyper_params_pso.c1) 
        self.spinbox_c2_pso.setValue(self.settings.hyper_params_pso.c2) 
        return

    def save_settings(self, pth="../../settings.ini"):
        if self.rbtn_manual.isChecked():
            self.settings.FLAGS_THETA0 = FLAGS_THETA0.MAMUAL.value
        if self.rbtn_pso.isChecked():
            self.settings.FLAGS_THETA0 = FLAGS_THETA0.MAMUAL.value
        if self.rbtn_epnp.isChecked():
            self.settings.FLAGS_THETA0 = FLAGS_THETA0.MAMUAL.value
        if self.rbtn_none.isChecked():
            self.settings.FLAGS_THETA0 = FLAGS_THETA0.MAMUAL.value

        if self.rbtn_adam.isChecked():
            self.settings.FLAGS_OPT = FLAGS_OPT.ADAM.value
        if self.rbtn_lm.isChecked():
            self.settings.FLAGS_OPT = FLAGS_OPT.LM.value

        if self.rbtn_correspond.isChecked():
            self.settings.FLAGS_POINTS2D = FLAGS_POINTS2D.CORRESPOND.value
        if self.rbtn_ellipse.isChecked():
            self.settings.FLAGS_POINTS2D = FLAGS_POINTS2D.ELLIPSE.value
        
        self.settings.hyper_params_adam.n_iters = self.spinbox_niters_adam.value()
        self.settings.hyper_params_adam.alpha   = self.spinbox_alpha_adam.value()
        self.settings.hyper_params_adam.beta1   = self.spinbox_beta1_adam.value()
        self.settings.hyper_params_adam.beta2   = self.spinbox_beta2_adam.value()

        self.settings.hyper_params_lm.n_iters = self.spinbox_niters_lm.value()
        self.settings.hyper_params_lm.alpha   = self.spinbox_alpha_lm.value()

        self.settings.hyper_params_pso.n_iters = self.spinbox_niters_pso.value()
        self.settings.hyper_params_pso.n_pops  = self.spinbox_npops_pso.value()
        self.settings.hyper_params_pso.w       = self.spinbox_w_pso.value()
        self.settings.hyper_params_pso.c1      = self.spinbox_c1_pso.value()
        self.settings.hyper_params_pso.c2      = self.spinbox_c2_pso.value()

        print("保存设置文件:", pth)
        with open(pth, "w") as f:
            json.dump(self.settings, f)
        return

    def new_settings(self):
        self.settings = EasyDict({})

        self.settings.FLAGS_CALIB    = FLAGS_CALIB.DLT.value
        self.settings.FLAGS_POINTS2D = FLAGS_POINTS2D.CORRESPOND.value
        self.settings.FLAGS_OPT      = FLAGS_OPT.ADAM.value
        self.settings.FLAGS_THETA0   = FLAGS_THETA0.MAMUAL.value

        # n_iters=1000, alpha=1E-3, beta1=0.9, beta2=0.999
        self.settings.hyper_params_adam = EasyDict({})
        self.settings.hyper_params_adam.n_iters = 1000
        self.settings.hyper_params_adam.alpha   = 1E-3
        self.settings.hyper_params_adam.beta1   = 0.9
        self.settings.hyper_params_adam.beta2   = 0.999

        # n_iters=1000, alpha=1E-3, beta1=0.9, beta2=0.999
        self.settings.hyper_params_lm = EasyDict({})
        self.settings.hyper_params_lm.n_iters = 1000
        self.settings.hyper_params_lm.alpha   = 1E-3
        
        # n_iters=1000, alpha=1E-3, beta1=0.9, beta2=0.999
        self.settings.hyper_params_pso = EasyDict({})
        self.settings.hyper_params_pso.n_iters = 1000
        self.settings.hyper_params_pso.n_pops  = 1000
        self.settings.hyper_params_pso.w       = 1E-3
        self.settings.hyper_params_pso.c1      = 0.9
        self.settings.hyper_params_pso.c2      = 0.999
        return


if  __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    widget = SettingsDialog(None)
    widget.new_settings()
    widget.load_settings()
    widget.save_settings()
    widget.show()
    sys.exit(app.exec_())
