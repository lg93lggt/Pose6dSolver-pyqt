
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\Li\Desktop\Pose6dSolver-pyqt\ui_files\DockGraphWidget.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(682, 440)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.dockGraphWidget = QtWidgets.QDockWidget(Form)
        self.dockGraphWidget.setFloating(False)
        self.dockGraphWidget.setFeatures(QtWidgets.QDockWidget.DockWidgetFloatable|QtWidgets.QDockWidget.DockWidgetMovable)
        self.dockGraphWidget.setObjectName("dockGraphWidget")
        self.dockWidgetContents = QtWidgets.QWidget()
        self.dockWidgetContents.setEnabled(True)
        self.dockWidgetContents.setObjectName("dockWidgetContents")
        self.layout_main = QtWidgets.QHBoxLayout(self.dockWidgetContents)
        self.layout_main.setObjectName("layout_main")
        self.groupbox_visualize = QtWidgets.QGroupBox(self.dockWidgetContents)
        self.groupbox_visualize.setObjectName("groupbox_visualize")
        self.layout_visualize = QtWidgets.QVBoxLayout(self.groupbox_visualize)
        self.layout_visualize.setObjectName("layout_visualize")
        self.graphics_view = QtWidgets.QGraphicsView(self.groupbox_visualize)
        self.graphics_view.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.graphics_view.setMouseTracking(True)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.NoBrush)
        self.graphics_view.setBackgroundBrush(brush)
        self.graphics_view.setObjectName("graphics_view")
        self.layout_visualize.addWidget(self.graphics_view)
        self.layout_coord_bar = QtWidgets.QHBoxLayout()
        self.layout_coord_bar.setObjectName("layout_coord_bar")
        self.label1 = QtWidgets.QLabel(self.groupbox_visualize)
        self.label1.setObjectName("label1")
        self.layout_coord_bar.addWidget(self.label1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.layout_coord_bar.addItem(spacerItem)
        self.label2 = QtWidgets.QLabel(self.groupbox_visualize)
        self.label2.setText("")
        self.label2.setObjectName("label2")
        self.layout_coord_bar.addWidget(self.label2)
        spacerItem1 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.layout_coord_bar.addItem(spacerItem1)
        self.label3 = QtWidgets.QLabel(self.groupbox_visualize)
        self.label3.setText("")
        self.label3.setObjectName("label3")
        self.layout_coord_bar.addWidget(self.label3)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.layout_coord_bar.addItem(spacerItem2)
        self.layout_visualize.addLayout(self.layout_coord_bar)
        self.layout_main.addWidget(self.groupbox_visualize)
        self.groupbox_points = QtWidgets.QGroupBox(self.dockWidgetContents)
        self.groupbox_points.setObjectName("groupbox_points")
        self.layout_points = QtWidgets.QVBoxLayout(self.groupbox_points)
        self.layout_points.setObjectName("layout_points")
        self.groupbox_show_points = QtWidgets.QGroupBox(self.groupbox_points)
        self.groupbox_show_points.setObjectName("groupbox_show_points")
        self.layout_show_points = QtWidgets.QHBoxLayout(self.groupbox_show_points)
        self.layout_show_points.setObjectName("layout_show_points")
        self.table_widget_show_points = QtWidgets.QTableWidget(self.groupbox_show_points)
        self.table_widget_show_points.setObjectName("tableWidgetShowPoints")
        self.table_widget_show_points.setColumnCount(0)
        self.table_widget_show_points.setRowCount(0)
        self.layout_show_points.addWidget(self.table_widget_show_points)
        self.layout_points.addWidget(self.groupbox_show_points)
        self.groupbox_choose_points = QtWidgets.QGroupBox(self.groupbox_points)
        self.groupbox_choose_points.setFlat(False)
        self.groupbox_choose_points.setCheckable(False)
        self.groupbox_choose_points.setObjectName("groupbox_choose_points")
        self.layout_choose_points = QtWidgets.QHBoxLayout(self.groupbox_choose_points)
        self.layout_choose_points.setObjectName("layout_choose_points")
        self.layout_points.addWidget(self.groupbox_choose_points)
        self.layout_main.addWidget(self.groupbox_points)
        self.dockGraphWidget.setWidget(self.dockWidgetContents)
        self.gridLayout.addWidget(self.dockGraphWidget, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.groupbox_visualize.setTitle(_translate("Form", "GroupBox"))
        self.label1.setText(_translate("Form", "像素坐标(u, v):"))
        self.groupbox_points.setTitle(_translate("Form", "点对:"))
        self.groupbox_show_points.setTitle(_translate("Form", "显示:"))
        self.groupbox_choose_points.setTitle(_translate("Form", "选择:"))


