
import os
import sys
import cv2
from  typing import *

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui     import *
from PyQt5.QtCore    import *
import numpy as np


sys.path.append("./src")
from ui import *
from core import *
from widgets import MainWindow

if __name__ == "__main__":

    import sys

    app = QtWidgets.QApplication(sys.argv)    
    main_window = MainWindow.MainWindow(debug=True)
    main_window.show()
    sys.exit(app.exec_())