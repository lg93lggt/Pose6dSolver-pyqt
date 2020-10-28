
import sys
from typing import *

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


sys.path.append("..")
from ui import *
from core import Visualizer, FileIO, geometry
from widgets import TableWidget


class DockGraphWidget(QWidget, Ui_DockGraphWidget.Ui_Form):
    # SigMouseMoveInItem              = pyqtSignal(float, float)
    # sig_mouse_lbtn_double_click_in_item = pyqtSignal(float, float)
    sig_sub_tabel_double_clicked  = pyqtSignal(str, str, int ,int)
    sig_choose_points2d_successed = pyqtSignal(str, str, dict, np.ndarray, np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.debug = parent.debug if parent else True
        self.img_raw  = np.array([[]])
        self.vis = Visualizer.Visualizer()
        
        self.n_objs = 0
        self.points2d_objs = {}
        self.points2d_chosen_tmp = []

        # 创建绘图场景
        self.scene = QGraphicsScene()  

        # 定义事件
        self.graphics_view.mouseMoveEvent        = self.event_mouse_move
        self.graphics_view.wheelEvent            = self.event_wheel_roll
        self.graphics_view.mouseDoubleClickEvent = self.event_mouse_double_click
        #self.graphics_view.mousePressEvent       = self.mouse_press
        return
    
    def init_sub_table_widgets(self, n_objs: int=1) -> None:
        n_objs_next = n_objs
        n_objs_prev = self.n_objs

        if n_objs_next > n_objs_prev:
            for i_obj in range(n_objs_next):
                if i_obj >= n_objs_prev: 

                    sub_table_widget = TableWidget.ObjectTableWidget(self)
                    sub_table_widget.sig_tabel_double_clicked.connect(self.solt_sub_table_widget_double_clicked)
                    sub_table_widget.sig_tabel_double_clicked.connect(self.solt_table_widget_show_points_refresh)
                    sub_table_widget.setObjectName("obj_{:d}".format(i_obj + 1))
                    layout_tab = QVBoxLayout()
                    layout_tab.addWidget(sub_table_widget)
                    sub_table_widget.setLayout(layout_tab)
                    sub_table_widget.setObjectName("obj_{}".format(i_obj + 1))
                    self.tab_widget_objs.addTab(sub_table_widget, "物体{}".format(i_obj + 1))
                    #self.groupbox_choose_points.setLayout(self.layout_choose_points)
        elif  n_objs_next < n_objs_prev:
            for i_obj in range(n_objs_next):
                if i_obj >= n_objs_next:
                    self.layout_choose_points.destroyed(self.sub_table_widgets[n_objs_next - 1])
        else:
            pass
        self.n_objs = n_objs
        return

    def init_img(self, img_input: np.ndarray) -> None:
        if len(img_input.shape) == 2:
            img_bgr = cv2.cvtColor(img_input, cv2.COLOR_GRAY2BGR)
        elif len(img_input.shape) == 3:
            img_bgr = img_input
        else:
            raise IndexError("图像读取失败.")
        self.img_raw  = img_bgr
        self.img_show = img_bgr.copy()

        self.scale  = 1
        self.dscale = 0.2

        self.item = self.imgbgr2item(self.img_show)

        self.scene.addItem(self.item)
        self.graphics_view.setScene(self.scene)  # 将场景添加至视图
        self.graphics_view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        return


    def imgbgr2item(self, imgbgr):
        img_rgb = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2RGB)
        qimg    = QtGui.QImage(img_rgb, img_rgb.shape[1], img_rgb.shape[0], QtGui.QImage.Format_RGB888)
        pix     = QtGui.QPixmap.fromImage(qimg)
        item    = QGraphicsPixmapItem(pix)  # 创建像素图元
        return item

    def _update(self):
        return

    def _update_img(self) -> None:
        # 重载图像控件
        [H_new, W_new] = [round(self.img_raw.shape[0] * self.scale), round(self.img_raw.shape[1] * self.scale)]
        self.img_show = cv2.resize(self.img_show, (W_new, H_new))
        self.scene.removeItem(self.item)
        self.item = self.imgbgr2item(self.img_show)
        self.scene.addItem(self.item)
        self.scene.setSceneRect(0,0, W_new, H_new)
        return


    def _update_table_widget_show_points(self) -> None:
        # 重载显点控件
        texts1 = []
        texts2 = []
        n_rows = 0
        for i_obj in range(self.n_objs):
            name_obj           = "obj_{:d}".format(i_obj + 1)
            sub_table_view     = self.get_sub_table_view(i_obj)
            points3d_from_chosen = sub_table_view.array_chosen
            points2d_from_image  = np.array([])
            n_rows_tmp         = points3d_from_chosen.shape[0]

            if name_obj in self.points2d_objs.keys():
                points2d_from_image = self.points2d_objs[name_obj]
                if points2d_from_image is None:
                    texts2.append("")
                    texts1.append("")
                    return 
            for i_row in range(n_rows_tmp):
                texts2.append(points3d_from_chosen[i_row])
                if len(points2d_from_image) == len(points3d_from_chosen):
                    texts1.append(points2d_from_image[i_row])
                else:
                    texts1.append("")
            texts2.append("")
            texts1.append("")
            n_rows += n_rows_tmp
            n_rows += 1
        
        # 设置子控件
        self.table_widget_show_points.setColumnCount(2)
        self.table_widget_show_points.setRowCount(n_rows)
        
        for i_row in range(n_rows):
            item1 = QTableWidgetItem(str(texts1[i_row]))
            item2 = QTableWidgetItem(str(texts2[i_row]))
            self.table_widget_show_points.setItem(i_row, 0, item1)
            self.table_widget_show_points.setItem(i_row, 1, item2)
        return

    def get_sub_table_view(self, obj: str or int) -> TableWidget.ObjectTableWidget:
        if isinstance(obj, int):
            obj = "obj_{}".format(obj + 1)
        sub_table_view = self.findChild(TableWidget.ObjectTableWidget, obj)
        return sub_table_view

    def draw_points2d(self) -> None:
        self.img_show = self.img_raw.copy()
        # 绘制已确定点
        for name_obj in self.points2d_objs.keys():
            if self.points2d_objs[name_obj] is None:
                continue
            for point in self.points2d_objs[name_obj]:
                self.img_show = cv2.circle(self.img_show, (point[0], point[1]), 1, (0, 255, 0), 1)
        # 绘制待确定点
        for point in self.points2d_chosen_tmp:
            self.img_show = cv2.circle(self.img_show, (point[0], point[1]), 1, (0, 0, 255), 1)
        return


    def set_data_sub_table_widgets(self, points3d: np.ndarray, i_obj: int) -> None:
        name = "obj_{:d}".format(i_obj + 1)
        sub_table_widget = self.get_sub_table_view(name)
        sub_table_widget.init_array(points3d)
        return

    def solt_mode_receive(self, mode: str):
        self.mode = mode

        if self.debug:
            print("[DEBUG]:\t<{}>  MODE SET <{}>".format(self.objectName(), mode))
        return

    def solt_sub_table_widget_double_clicked(self, name_sub_table_widget: str, i_row: int, i_col: int):
        self.sig_sub_tabel_double_clicked.emit(self.objectName(), name_sub_table_widget, i_row, i_col)

        if self.debug:
            print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_sub_tabel_double_clicked.signal))
        return

    def solt_table_widget_show_points_refresh(self, name_obj: str, i_row: int, i_col: int):
        self._update_table_widget_show_points()
        return

    def slot_draw_calib_result(self, i_cam: int, camera_pars: Dict):
        if self.objectName() == "cam_{:d}".format(i_cam + 1):
            print(self.objectName(), i_cam, "\n", camera_pars["intrin"])
            self.camera_pars = camera_pars
            self.img_show = self.img_raw.copy()
            self.vis.draw_axis3d(self.img_show, self.camera_pars)
            for [name_obj, points2d] in self.points2d_objs.items():
                self.vis.draw_points2d(self.img_show, points2d, 10)
                sub_tabel_widget = self.get_sub_table_view(name_obj)
                self.vis.draw_points3d(self.img_show, sub_tabel_widget.array, np.zeros(6), self.camera_pars)
            self._update_img()
        return

    def slot_draw_solve_result(self, i_obj: int, i_cam: int, theta: np.ndarray, camera_pars: Dict):
        if self.objectName() == "cam_{:d}".format(i_cam + 1):
            self.camera_pars = camera_pars
            self.img_show = self.img_raw.copy()
            points2d = self.points2d_objs["obj_{:d}".format(i_obj + 1)]
            self.vis.draw_points2d(self.img_show, points2d, 1)
            sub_tabel_widget = self.get_sub_table_view(i_obj)
            points3d =  sub_tabel_widget.array
            self.vis.draw_axis3d(self.img_show, self.camera_pars)
            self.vis.draw_backbone3d(self.img_show, points3d, theta, self.camera_pars)
            self.vis.draw_points3d_with_texts(self.img_show, points3d, theta, self.camera_pars)
            # self.vis.draw_model3d(self.img_show, model, theta, self.camera_pars)
            self._update_img()
        return
        
    def slot_draw_theta0(self, name_obj: str, i_cam: int, theta: np.ndarray):
        theta = geometry.rtvec_degree2rad(theta)
        print("draw :", theta)
        if self.objectName() == "cam_{:d}".format(i_cam + 1):
            self.img_show = self.img_raw.copy()
            
            sub_tabel_widget = self.get_sub_table_view(name_obj)
            points3d =  sub_tabel_widget.array
            self.vis.draw_points3d_with_texts(self.img_show, points3d, theta, self.camera_pars)
            self.vis.draw_backbone3d(self.img_show, points3d, theta, self.camera_pars)
            self.vis.draw_axis3d(self.img_show, self.camera_pars)
            # 先不画模型
            # if self.parent().parent().parent().parent():
            #     main = self.parent().parent().parent().parent()
            #     for i_model in range(main.fio.struct.solve.n_models):
            #         self.vis.draw_model3d(self.img_show, main.fio.load_model("solve", name_obj), theta, self.camera_pars)
            self._update_img()
        return


    ## 事件----------------------------------------------------------------------------------------------##
    def event_mouse_move(self, evt: QMouseEvent) -> None:
        """
            鼠标移动事件
        """
        pt_scene = self.graphics_view.mapToScene(evt.x(), evt.y())  #把view坐标转换为场景坐标
        if self.scene:
            item = self.scene.itemAt(pt_scene, self.graphics_view.transform())  #在场景某点寻找图元--最上面的图元
            if item != None:
                pt_item = item.mapFromScene(pt_scene)  #把场景坐标转换为图元坐标
                [u_raw, v_raw] = [round(pt_item.x() / self.scale), round(pt_item.y() / self.scale)]
                self.label2.setText(str(u_raw))
                self.label3.setText(str(v_raw))
        return

    def event_wheel_roll(self, evt: QWheelEvent):
        """
            滚轮事件
        """
        pt_scene = self.graphics_view.mapToScene(evt.x(), evt.y())  #把view坐标转换为场景坐标
        item = self.scene.itemAt(pt_scene, self.graphics_view.transform())  #在场景某点寻找图元--最上面的图元
        if item != None:
            pt_item = item.mapFromScene(pt_scene)  #把场景坐标转换为图元坐标
            if evt.angleDelta().y() > 0: # 放大图片
                self.scale += self.dscale
                [u_prev, v_prev] = [round(pt_item.x()), round(pt_item.y())]
                self._update_img()
                #self.graphicsView.centerOn(u, v)
                
            elif evt.angleDelta().y() < 0: # 缩小图片
                self.scale -= self.dscale
                if self.scale < 1:
                    self.scale = 1
                self._update_img()
            return

    def event_mouse_double_click(self, evt: QMouseEvent) -> None:
        """
            鼠标双击事件
        """
        if evt.button() == Qt.LeftButton: # 左键双击选点
            pt_scene = self.graphics_view.mapToScene(evt.x(), evt.y()) #把view坐标转换为场景坐标
            item = self.scene.itemAt(pt_scene,self.graphics_view.transform()) #在场景某点寻找图元--最上面的图元
            if item != None:
                pt_item = item.mapFromScene(pt_scene) #把场景坐标转换为图元坐标
                [u_raw, v_raw] = [round(pt_item.x() / self.scale), round(pt_item.y() / self.scale)]
                [u_raw, v_raw] = [int(u_raw), int(v_raw)]
                point2d = np.array([u_raw, v_raw], dtype=int)
                self.points2d_chosen_tmp.append(point2d)
                n_points = len(self.points2d_chosen_tmp)
                self.draw_points2d()
                self._update_img()
                print("点 {:d}:\t".format(n_points), point2d)
            return
        elif evt.button() == Qt.RightButton: # 左键双击撤销
            pt_scene = self.graphics_view.mapToScene(evt.x(), evt.y())  #把view坐标转换为场景坐标
            item = self.scene.itemAt(pt_scene,self.graphics_view.transform())  #在场景某点寻找图元--最上面的图元
            if item != None:
                n_points = len(self.points2d_chosen_tmp)
                if n_points > 0:
                    self.points2d_chosen_tmp.pop()
                    self.draw_points2d()
                    self._update_img()
                    print("删除点 {:d}.".format(n_points))
            return
    
    def keyPressEvent(self, evt: QtGui.QKeyEvent) -> None:
        """
            键盘事件
        """
        texts_premit = ["{:d}".format(x + 1) for x in list(range((self.n_objs)))]
        if evt.text() in texts_premit:
            i_obj          = int(evt.text()) - 1
            name_obj       = "obj_{:d}".format(i_obj + 1)
            sub_table_view = self.get_sub_table_view(i_obj)
            n_points_tmp   = len(self.points2d_chosen_tmp)
            n_points_obj   = sub_table_view.array_chosen.shape[0]
            if n_points_tmp == n_points_obj:
                self.points2d_objs[name_obj] = self.points2d_chosen_tmp
                self.points2d_chosen_tmp           = []
                self.sig_choose_points2d_successed.emit(
                    self.objectName(), 
                    name_obj, 
                    self.points2d_objs, 
                    self.get_sub_table_view(name_obj).array_chosen, 
                    self.get_sub_table_view(name_obj).indexes_chosen
                )
                print("物体{}:\t".format(name_obj, self.points2d_objs[name_obj]))
                
                if self.debug:
                    print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_choose_points2d_successed.signal))
            else:
                print("点数不正确!{}/{}".format(n_points_tmp, n_points_obj))
        self._update_table_widget_show_points()
        return


if  __name__ == "__main__": 
    import sys

    app = QtWidgets.QApplication(sys.argv)
    widget = DockGraphWidget(None)
    img = cv2.imread("../../姿态测量/images_calib/cam_1/scene_1.png")
    pts_n_objs =[ np.loadtxt("../../姿态测量/points3d_calib/obj_1.txt")]
    widget.init_img(img)
    widget.init_sub_table_widgets(3)
    widget.show()
    sys.exit(app.exec_())
