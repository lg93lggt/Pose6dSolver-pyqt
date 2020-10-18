# -*- coding: utf-8 -*-
# Created by: PyQt5 UI code generator 5.15.1

import sys
import  os

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui     import *
from PyQt5.QtCore    import *
import cv2
import numpy as np

sys.path.append("..")
from ui import Ui_ScenesTableWidget



class ScenesTableWidget(QWidget, Ui_ScenesTableWidget.Ui_Form):
    sig_tabel_double_clicked = pyqtSignal(str, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.debug = parent.debug if parent else True
        self.table_widget.setEditTriggers(QTableView.NoEditTriggers)  # 不可编辑
        self.table_widget.setSelectionBehavior(QAbstractItemView.SelectRows);  # 设置只有行选中
        return

    def set_shape(self, n_rows=0, n_cols=0):
        self.table_widget.setWordWrap(True)
        self.table_widget.setCornerButtonEnabled(True)

        self.table_widget.setRowCount(n_rows)
        self.table_widget.setColumnCount(n_cols)

        for i_row in range(n_rows):
            item = QtWidgets.QTableWidgetItem()
            self.table_widget.setVerticalHeaderItem(i_row, item)
        for i_col in range(n_cols):
            item = QtWidgets.QTableWidgetItem()
            if i_col < n_cols - 3:
                item.setCheckState(QtCore.Qt.Unchecked)
            self.table_widget.setVerticalHeaderItem(i_col, item)

        self.horizontalLayout.addWidget(self.table_widget)
        self.setLayout(self.horizontalLayout)
        return

    def set_texts(self, texts_rows=None, texts_cols=None):
        if texts_rows:
            self.table_widget.setVerticalHeaderLabels(texts_rows)
        if texts_cols:
            self.table_widget.setHorizontalHeaderLabels(texts_cols)
        return

    def set_checkbox(self, i_row: int, i_col: int, checked: bool=False):
        ck = QCheckBox()
        ck.setChecked(checked)
        hlayout = QHBoxLayout()
        hlayout.setAlignment(Qt.AlignHCenter)
        hlayout.addWidget(ck)

        w = QWidget()
        w.setLayout(hlayout)
        self.table_widget.setCellWidget(i_row, i_col, w)
        return

    def set_checkboxes(self, i_row: int=None, i_col: int=None, checked: bool=False):
        n_cols = self.table_widget.columnCount()
        n_rows = self.table_widget.rowCount()
        
        if not(i_row is None):
            for j_col in range(n_cols):
                self.set_checkbox(i_row, j_col, checked)
                
        if not(i_col is None):
            self.table_widget.setColumnWidth(i_col, 50)
            for j_row in range(n_rows):
                self.set_checkbox(j_row, i_col, checked)
        return

    def set_datas_all(self, datas):
        for i in range(len(datas)):
            for j in range(len(datas[i])):
                item = QTableWidgetItem(str(datas[i][j]))
                self.table_widget.setItem(i, j, item)
        return

    def set_datas_col(self, datas, icol):
        for i_row in range(len(datas)):
            item = QTableWidgetItem(str(datas[i_row]))
            self.table_widget.setItem(i_row, icol, item)
        return

    def set_datas_row(self, datas, i_row):
        for i_col in range(len(datas)):
            item = QTableWidgetItem(str(datas[i_col]))
            self.table_widget.setItem(i_row, i_col, item)
        return
        
    @pyqtSlot(int, int)
    def on_tableWidget_cellDoubleClicked(self, i_row: int, i_col: int):
        self.sig_tabel_double_clicked.emit(self.objectName(), i_row, i_col)
        return

    def solt_mode_receive(self, mode: str):
        self.mode = mode
        print("{} mode = {}".format(self.objectName(), mode))
        return

class ObjectTableWidget(ScenesTableWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.array = None
        self.indexes_chosen = None
        return

    def init_array(self, array: np.ndarray) -> None:
        shape = array.shape
        self.set_shape(shape[0], 1)
        self.set_array(array)
        #self.set_checkboxes(i_col=1, checked=True)
        self.state_chosen  = np.ones(shape[0], dtype=bool)
        self.indexes_chosen = np.arange(shape[0])
        return

    def set_array(self, array: np.ndarray) -> None:
        for i_row in range(len(array)):
            item = QTableWidgetItem(str(array[i_row]))
            item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            item.setCheckState(Qt.Checked)
            self.table_widget.setItem(i_row, 0, item)
        self.array = array
        self.array_chosen = self.array.copy()
        return

    def _update_chosen_idexes(self):
        n_rows = self.table_widget.rowCount()
        for i_row in range(n_rows):
            check_state = self.table_widget.item(i_row, 0).checkState()
            if check_state == Qt.Checked:
                self.state_chosen[i_row] = True
            elif check_state == Qt.Unchecked:
                self.state_chosen[i_row] = False
        self.indexes_chosen = np.where(self.state_chosen==True)[0]
        self.array_chosen = self.array[self.indexes_chosen]
        return

    @pyqtSlot(int, int)
    def on_tableWidget_cellDoubleClicked(self, i_row: int, i_col: int):
        check_state = self.table_widget.item(i_row, 0).checkState()
        if check_state == Qt.Checked:
            self.table_widget.item(i_row, 0).setCheckState(Qt.Unchecked)
        elif check_state == Qt.Unchecked:
            self.table_widget.item(i_row, 0).setCheckState(Qt.Checked)
        self._update_chosen_idexes()
        self.sig_tabel_double_clicked.emit(self.objectName(), i_row, i_col)
        return


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    widget = ObjectTableWidget()
    widget.init_array(np.eye(4))
    widget.show()
    sys.exit(app.exec_())
