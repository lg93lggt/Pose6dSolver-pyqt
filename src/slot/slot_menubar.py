import sys

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui     import *
from PyQt5.QtCore    import *

sys.path.append("..")
from widgets import NewProject, OpenProject, EditProject 

def slot_creat_new_project():
    return widget

def open_project():
    widget = OpenProject.OpenProjectDialog()
    #widget.setObjectName("openProjectDialog")
    return widget

def edit_project():
    widget = EditProject.EditProject()
    return widget