
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1075, 908)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupbox_visualize = QtWidgets.QGroupBox(Form)
        self.groupbox_visualize.setObjectName("groupbox_visualize")
        self.layout = QtWidgets.QGridLayout(self.groupbox_visualize)
        self.layout.setObjectName("layout")
        self.horizontalLayout.addWidget(self.groupbox_visualize)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.groupbox_visualize.setTitle(_translate("Form", "可视化区域:"))