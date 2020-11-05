# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/veily/桌面/Pose6dSolver-pyqt/ui_files/ManualPoseWidget_revamp.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(250, 155)
        Form.setMaximumSize(QtCore.QSize(250, 16777215))
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_rvec = QtWidgets.QLabel(Form)
        self.label_rvec.setMaximumSize(QtCore.QSize(16777215, 50))
        self.label_rvec.setWordWrap(True)
        self.label_rvec.setObjectName("label_rvec")
        self.gridLayout.addWidget(self.label_rvec, 0, 1, 1, 1)
        self.label_tvec = QtWidgets.QLabel(Form)
        self.label_tvec.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_tvec.setObjectName("label_tvec")
        self.gridLayout.addWidget(self.label_tvec, 0, 2, 1, 1)
        self.label_x = QtWidgets.QLabel(Form)
        self.label_x.setMinimumSize(QtCore.QSize(30, 0))
        self.label_x.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_x.setAlignment(QtCore.Qt.AlignCenter)
        self.label_x.setObjectName("label_x")
        self.gridLayout.addWidget(self.label_x, 1, 0, 1, 1)
        self.double_spin_box_rx = QtWidgets.QDoubleSpinBox(Form)
        self.double_spin_box_rx.setDecimals(3)
        self.double_spin_box_rx.setMinimum(-179.99)
        self.double_spin_box_rx.setMaximum(180.0)
        self.double_spin_box_rx.setSingleStep(0.1)
        self.double_spin_box_rx.setProperty("value", 0.0)
        self.double_spin_box_rx.setObjectName("double_spin_box_rx")
        self.gridLayout.addWidget(self.double_spin_box_rx, 1, 1, 1, 1)
        self.double_spin_box_tx = QtWidgets.QDoubleSpinBox(Form)
        self.double_spin_box_tx.setDecimals(3)
        self.double_spin_box_tx.setMinimum(-99.99)
        self.double_spin_box_tx.setSingleStep(0.01)
        self.double_spin_box_tx.setObjectName("double_spin_box_tx")
        self.gridLayout.addWidget(self.double_spin_box_tx, 1, 2, 1, 1)
        self.label_y = QtWidgets.QLabel(Form)
        self.label_y.setMinimumSize(QtCore.QSize(30, 0))
        self.label_y.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_y.setAlignment(QtCore.Qt.AlignCenter)
        self.label_y.setObjectName("label_y")
        self.gridLayout.addWidget(self.label_y, 2, 0, 1, 1)
        self.double_spin_box_ry = QtWidgets.QDoubleSpinBox(Form)
        self.double_spin_box_ry.setWrapping(False)
        self.double_spin_box_ry.setDecimals(3)
        self.double_spin_box_ry.setMinimum(-179.99)
        self.double_spin_box_ry.setMaximum(180.0)
        self.double_spin_box_ry.setSingleStep(0.1)
        self.double_spin_box_ry.setObjectName("double_spin_box_ry")
        self.gridLayout.addWidget(self.double_spin_box_ry, 2, 1, 1, 1)
        self.double_spin_box_ty = QtWidgets.QDoubleSpinBox(Form)
        self.double_spin_box_ty.setDecimals(3)
        self.double_spin_box_ty.setMinimum(-99.99)
        self.double_spin_box_ty.setSingleStep(0.01)
        self.double_spin_box_ty.setObjectName("double_spin_box_ty")
        self.gridLayout.addWidget(self.double_spin_box_ty, 2, 2, 1, 1)
        self.label_z = QtWidgets.QLabel(Form)
        self.label_z.setMinimumSize(QtCore.QSize(30, 0))
        self.label_z.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_z.setAlignment(QtCore.Qt.AlignCenter)
        self.label_z.setObjectName("label_z")
        self.gridLayout.addWidget(self.label_z, 3, 0, 1, 1)
        self.double_spin_box_rz = QtWidgets.QDoubleSpinBox(Form)
        self.double_spin_box_rz.setDecimals(3)
        self.double_spin_box_rz.setMinimum(-179.99)
        self.double_spin_box_rz.setMaximum(180.0)
        self.double_spin_box_rz.setSingleStep(0.1)
        self.double_spin_box_rz.setObjectName("double_spin_box_rz")
        self.gridLayout.addWidget(self.double_spin_box_rz, 3, 1, 1, 1)
        self.double_spin_box_tz = QtWidgets.QDoubleSpinBox(Form)
        self.double_spin_box_tz.setDecimals(3)
        self.double_spin_box_tz.setMinimum(-99.99)
        self.double_spin_box_tz.setSingleStep(0.01)
        self.double_spin_box_tz.setObjectName("double_spin_box_tz")
        self.gridLayout.addWidget(self.double_spin_box_tz, 3, 2, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_rvec.setText(_translate("Form", "旋转向量(degree)  :"))
        self.label_tvec.setText(_translate("Form", "平移向量(m):"))
        self.label_x.setText(_translate("Form", "X"))
        self.label_y.setText(_translate("Form", "Y"))
        self.label_z.setText(_translate("Form", "Z"))

