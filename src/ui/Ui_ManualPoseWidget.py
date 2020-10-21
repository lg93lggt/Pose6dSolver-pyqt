
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\Li\Desktop\Pose6dSolver-pyqt\ui_files\ManualPoseWidget.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(437, 131)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        self.layout_retvec = QtWidgets.QGridLayout()
        self.layout_retvec.setObjectName("layout_retvec")
        self.line_rz = QtWidgets.QLineEdit(Form)
        self.line_rz.setMaximumSize(QtCore.QSize(16777215, 20))
        self.line_rz.setObjectName("line_rz")
        self.layout_retvec.addWidget(self.line_rz, 1, 4, 1, 1)
        self.line_ry = QtWidgets.QLineEdit(Form)
        self.line_ry.setMaximumSize(QtCore.QSize(16777215, 20))
        self.line_ry.setObjectName("line_ry")
        self.layout_retvec.addWidget(self.line_ry, 1, 3, 1, 1)
        self.line_rx = QtWidgets.QLineEdit(Form)
        self.line_rx.setMaximumSize(QtCore.QSize(16777215, 20))
        self.line_rx.setObjectName("line_rx")
        self.layout_retvec.addWidget(self.line_rx, 1, 2, 1, 1)
        self.line_tz = QtWidgets.QLineEdit(Form)
        self.line_tz.setMaximumSize(QtCore.QSize(16777215, 20))
        self.line_tz.setObjectName("line_tz")
        self.layout_retvec.addWidget(self.line_tz, 2, 4, 1, 1)
        self.line_tx = QtWidgets.QLineEdit(Form)
        self.line_tx.setMaximumSize(QtCore.QSize(16777215, 20))
        self.line_tx.setObjectName("line_tx")
        self.layout_retvec.addWidget(self.line_tx, 2, 2, 1, 1)
        self.line_ty = QtWidgets.QLineEdit(Form)
        self.line_ty.setMaximumSize(QtCore.QSize(16777215, 20))
        self.line_ty.setObjectName("line_ty")
        self.layout_retvec.addWidget(self.line_ty, 2, 3, 1, 1)
        self.label_tvec = QtWidgets.QLabel(Form)
        self.label_tvec.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_tvec.setObjectName("label_tvec")
        self.layout_retvec.addWidget(self.label_tvec, 2, 1, 1, 1)
        self.label_rvec = QtWidgets.QLabel(Form)
        self.label_rvec.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_rvec.setObjectName("label_rvec")
        self.layout_retvec.addWidget(self.label_rvec, 1, 1, 1, 1)
        self.label_y = QtWidgets.QLabel(Form)
        self.label_y.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_y.setAlignment(QtCore.Qt.AlignCenter)
        self.label_y.setObjectName("label_y")
        self.layout_retvec.addWidget(self.label_y, 0, 3, 1, 1)
        self.label_z = QtWidgets.QLabel(Form)
        self.label_z.setMinimumSize(QtCore.QSize(0, 0))
        self.label_z.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_z.setAlignment(QtCore.Qt.AlignCenter)
        self.label_z.setObjectName("label_z")
        self.layout_retvec.addWidget(self.label_z, 0, 4, 1, 1)
        self.label_x = QtWidgets.QLabel(Form)
        self.label_x.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_x.setAlignment(QtCore.Qt.AlignCenter)
        self.label_x.setObjectName("label_x")
        self.layout_retvec.addWidget(self.label_x, 0, 2, 1, 1)
        self.verticalLayout.addLayout(self.layout_retvec)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_rvec.setText(_translate("Form", "旋转向量(degree):"))
        self.label_tvec.setText(_translate("Form", "平移向量(m):"))
        self.label_y.setText(_translate("Form", "Y"))
        self.label_z.setText(_translate("Form", "Z"))
        self.label_x.setText(_translate("Form", "X"))


