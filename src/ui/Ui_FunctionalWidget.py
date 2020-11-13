

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\Li\Desktop\Pose6dSolver-pyqt\ui_files\FunctionalWidget.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(318, 545)
        Form.setMaximumSize(QtCore.QSize(320, 16777215))
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.btn_run = QtWidgets.QPushButton(Form)
        self.btn_run.setObjectName("btn_run")
        self.gridLayout.addWidget(self.btn_run, 0, 1, 1, 1)
        self.btn_save = QtWidgets.QPushButton(Form)
        self.btn_save.setObjectName("btn_save")
        self.gridLayout.addWidget(self.btn_save, 0, 0, 1, 1)
        self.groupbox_visualize_sets = QtWidgets.QGroupBox(Form)
        self.groupbox_visualize_sets.setObjectName("groupbox_visualize_sets")
        self.layout_visulize_sets = QtWidgets.QGridLayout(self.groupbox_visualize_sets)
        self.layout_visulize_sets.setObjectName("layout_visulize_sets")
        self.rbtn_points3d = QtWidgets.QCheckBox(self.groupbox_visualize_sets)
        self.rbtn_points3d.setChecked(True)
        self.rbtn_points3d.setObjectName("rbtn_points3d")
        self.layout_visulize_sets.addWidget(self.rbtn_points3d, 0, 0, 1, 1)
        self.rbtn_backbone = QtWidgets.QCheckBox(self.groupbox_visualize_sets)
        self.rbtn_backbone.setChecked(True)
        self.rbtn_backbone.setObjectName("rbtn_backbone")
        self.layout_visulize_sets.addWidget(self.rbtn_backbone, 1, 0, 1, 1)
        self.rbtn_points2d = QtWidgets.QCheckBox(self.groupbox_visualize_sets)
        self.rbtn_points2d.setChecked(True)
        self.rbtn_points2d.setObjectName("rbtn_points2d")
        self.layout_visulize_sets.addWidget(self.rbtn_points2d, 2, 0, 1, 1)
        self.rbtn_axis = QtWidgets.QCheckBox(self.groupbox_visualize_sets)
        self.rbtn_axis.setTabletTracking(False)
        self.rbtn_axis.setChecked(True)
        self.rbtn_axis.setAutoRepeat(False)
        self.rbtn_axis.setObjectName("rbtn_axis")
        self.layout_visulize_sets.addWidget(self.rbtn_axis, 3, 0, 1, 1)
        self.rbtn_model = QtWidgets.QCheckBox(self.groupbox_visualize_sets)
        self.rbtn_model.setChecked(True)
        self.rbtn_model.setObjectName("rbtn_model")
        self.layout_visulize_sets.addWidget(self.rbtn_model, 4, 0, 1, 1)
        self.gridLayout.addWidget(self.groupbox_visualize_sets, 5, 0, 1, 2)
        self.groupbox_tab = QtWidgets.QGroupBox(Form)
        self.groupbox_tab.setMinimumSize(QtCore.QSize(300, 200))
        self.groupbox_tab.setMaximumSize(QtCore.QSize(300, 200))
        self.groupbox_tab.setObjectName("groupbox_tab")
        self.layout_tab = QtWidgets.QGridLayout(self.groupbox_tab)
        self.layout_tab.setObjectName("layout_tab")
        self.tab_widget_objs = QtWidgets.QTabWidget(self.groupbox_tab)
        self.tab_widget_objs.setTabPosition(QtWidgets.QTabWidget.North)
        self.tab_widget_objs.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tab_widget_objs.setElideMode(QtCore.Qt.ElideLeft)
        self.tab_widget_objs.setDocumentMode(True)
        self.tab_widget_objs.setTabsClosable(False)
        self.tab_widget_objs.setTabBarAutoHide(False)
        self.tab_widget_objs.setObjectName("tab_widget_objs")
        self.tab1 = QtWidgets.QWidget()
        self.tab1.setObjectName("tab1")
        self.layout_subtab_1 = QtWidgets.QGridLayout(self.tab1)
        self.layout_subtab_1.setObjectName("layout_subtab_1")
        self.tab_widget_objs.addTab(self.tab1, "")
        self.layout_tab.addWidget(self.tab_widget_objs, 0, 2, 1, 1)
        self.gridLayout.addWidget(self.groupbox_tab, 2, 0, 1, 2)
        self.groupbox_outprint = QtWidgets.QGroupBox(Form)
        self.groupbox_outprint.setObjectName("groupbox_outprint")
        self.layou_text_edit = QtWidgets.QVBoxLayout(self.groupbox_outprint)
        self.layou_text_edit.setObjectName("layou_text_edit")
        self.text_edit_outprint = QtWidgets.QPlainTextEdit(self.groupbox_outprint)
        self.text_edit_outprint.setObjectName("text_edit_outprint")
        self.layou_text_edit.addWidget(self.text_edit_outprint)
        self.gridLayout.addWidget(self.groupbox_outprint, 3, 0, 2, 2)

        self.retranslateUi(Form)
        self.tab_widget_objs.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.btn_run.setText(_translate("Form", "运行"))
        self.btn_save.setText(_translate("Form", "保存关键点"))
        self.groupbox_visualize_sets.setTitle(_translate("Form", "可视化选项:"))
        self.rbtn_points3d.setText(_translate("Form", "关键点"))
        self.rbtn_backbone.setText(_translate("Form", "骨架"))
        self.rbtn_points2d.setText(_translate("Form", "标记点"))
        self.rbtn_axis.setText(_translate("Form", "坐标轴"))
        self.rbtn_model.setText(_translate("Form", "模型"))
        self.groupbox_tab.setTitle(_translate("Form", "手动调节:"))
        self.tab_widget_objs.setTabText(self.tab_widget_objs.indexOf(self.tab1), _translate("Form", "物体1"))
        self.groupbox_outprint.setTitle(_translate("Form", "命令行:"))


