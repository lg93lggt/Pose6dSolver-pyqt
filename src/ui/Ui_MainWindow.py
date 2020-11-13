# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\Li\Desktop\Pose6dSolver-pyqt\ui_files\MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.NonModal)
        MainWindow.resize(704, 532)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout1 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout1.setContentsMargins(8, 8, 8, 8)
        self.horizontalLayout1.setSpacing(5)
        self.horizontalLayout1.setObjectName("horizontalLayout1")
        self.layout_main = QtWidgets.QGridLayout()
        self.layout_main.setSpacing(5)
        self.layout_main.setObjectName("layout_main")
        self.horizontalLayout1.addLayout(self.layout_main)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 704, 23))
        self.menubar.setObjectName("menubar")
        self.menu1 = QtWidgets.QMenu(self.menubar)
        self.menu1.setObjectName("menu1")
        self.menu2 = QtWidgets.QMenu(self.menubar)
        self.menu2.setObjectName("menu2")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.action_new_project = QtWidgets.QAction(MainWindow)
        self.action_new_project.setObjectName("action_new_project")
        self.action_open_project = QtWidgets.QAction(MainWindow)
        self.action_open_project.setObjectName("action_open_project")
        self.action_calib = QtWidgets.QAction(MainWindow)
        self.action_calib.setObjectName("action_calib")
        self.action_edit_project = QtWidgets.QAction(MainWindow)
        self.action_edit_project.setObjectName("action_edit_project")
        self.action_solve = QtWidgets.QAction(MainWindow)
        self.action_solve.setObjectName("action_solve")
        self.action_video2images = QtWidgets.QAction(MainWindow)
        self.action_video2images.setObjectName("action_video2images")
        self.action_settings = QtWidgets.QAction(MainWindow)
        self.action_settings.setObjectName("action_settings")
        self.menu1.addAction(self.action_new_project)
        self.menu1.addAction(self.action_open_project)
        self.menu1.addSeparator()
        self.menu1.addAction(self.action_edit_project)
        self.menu1.addSeparator()
        self.menu1.addAction(self.action_settings)
        self.menu2.addAction(self.action_calib)
        self.menu2.addSeparator()
        self.menu2.addAction(self.action_solve)
        self.menu.addAction(self.action_video2images)
        self.menubar.addAction(self.menu1.menuAction())
        self.menubar.addAction(self.menu2.menuAction())
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Pose6dSolver"))
        self.menu1.setTitle(_translate("MainWindow", "文件"))
        self.menu2.setTitle(_translate("MainWindow", "功能"))
        self.menu.setTitle(_translate("MainWindow", "工具"))
        self.action_new_project.setText(_translate("MainWindow", "新建工程"))
        self.action_open_project.setText(_translate("MainWindow", "打开工程"))
        self.action_calib.setText(_translate("MainWindow", "标定"))
        self.action_edit_project.setText(_translate("MainWindow", "编辑工程"))
        self.action_solve.setText(_translate("MainWindow", "测量"))
        self.action_video2images.setText(_translate("MainWindow", "视频分割"))
        self.action_settings.setText(_translate("MainWindow", "参数设置"))

