
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
from core import Visualizer, FileIO, geometry, Conic
from widgets import TableWidget
from core.geometry import pose_to_rtvec


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

        self.n_objs = 0
        self.points2d_objs = {}
        self.points2d_chosen_tmp = []

        # 创建绘图场景
        self.scene = QGraphicsScene()
        self.scale  = 1

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
                    # layout_tab = QVBoxLayout()
                    #sub_table_widget.horizontalLayout.addWidget(sub_table_widget)
                    #sub_table_widget.horizontalLayout(layout_tab)
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
        self.img_raw    = img_bgr
        self.img_modify = img_bgr.copy()
        self.img_show   = img_bgr.copy()

        self.scale  = 1
        self.dscale = 0.2

        if len(self.scene.items()) > 0:
            self.scene.removeItem(self.item)

        self.item = self.imgbgr2item(self.img_show)
        
        self.scene.addItem(self.item)
        self.graphics_view.setScene(self.scene)  # 将场景添加至视图
        self.graphics_view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        # if self.debug:
        #     print("n items = {}".format(len(self.scene.items())))
        return

    def _update_img(self) -> None:
        # 重载图像控件
        [H_new, W_new] = [round(self.img_raw.shape[0] * self.scale), round(self.img_raw.shape[1] * self.scale)]
        self.img_show = cv2.resize(self.img_modify.copy(), (W_new, H_new), cv2.INTER_CUBIC)
        self.scene.removeItem(self.item)
        self.item = self.imgbgr2item(self.img_show)
        self.scene.addItem(self.item)
        self.scene.setSceneRect(0, 0, W_new, H_new)
        return

    def imgbgr2item(self, imgbgr):
        img_rgb = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2RGB)
        qimg    = QtGui.QImage(img_rgb, img_rgb.shape[1], img_rgb.shape[0], QtGui.QImage.Format_RGB888)
        pix     = QtGui.QPixmap.fromImage(qimg)
        item    = QGraphicsPixmapItem(pix)  # 创建像素图元
        return item

    def _init_table_widget_show_points(self) -> None:
        # 重载显点控件
        i_cam = int(self.objectName().split("_")[1]) - 1
        texts1 = []
        texts2 = []
        for i_obj in range(self.n_objs):
            name_obj        = "obj_{:d}".format(i_obj + 1)
            sub_table_view  = self.get_sub_table_view(i_obj)
            indexes3d       = self.window().objs[i_obj].views[i_cam].indexes3d
            points3d        = self.window().objs[i_obj].points3d
            points3d_chosen = points3d[indexes3d]
            points2d        = self.window().objs[i_obj].views[i_cam].points2d

            if points2d is None:
                texts2.append("")
                texts1.append("")
                return
            else:
                n_pts = points2d.shape[0]
                for i_pt in range(n_pts):
                    texts1.append(points2d[i_pt])
                    if i_pt < points3d_chosen.shape[0]:
                        texts2.append(points3d_chosen[i_pt])
                    else:
                        texts2.append("")
                texts2.append("")
                texts1.append("")

        # 设置子控件
        n_rows = len(texts1)
        self.table_widget_show_points.setColumnCount(2)
        self.table_widget_show_points.setRowCount(n_rows)

        for i_row in range(n_rows):
            item1 = QTableWidgetItem(str(texts1[i_row]))
            item2 = QTableWidgetItem(str(texts2[i_row]))
            self.table_widget_show_points.setItem(i_row, 0, item1)
            self.table_widget_show_points.setItem(i_row, 1, item2)
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
            print("[DEBUG]: {}".format(self.objectName()))
        return

    def solt_table_widget_show_points_refresh(self, name_obj: str, i_row: int, i_col: int):
        print("[DEBUG]: {}".format(self.objectName()))
        self._update_table_widget_show_points()
        return

    def draw_points2d(self) -> None:
        # self.img_modify = self.img_raw.copy() # 重绘
        # 绘制已确定点
        for name_obj in self.points2d_objs.keys():
            if self.points2d_objs[name_obj] is None:
                continue
            for point in self.points2d_objs[name_obj]:
                self.img_modify = cv2.circle(self.img_modify, (point[0], point[1]), 1, (0, 255, 0), 1)
        # 绘制待确定点
        for point in self.points2d_chosen_tmp:
            self.img_modify = cv2.circle(self.img_modify, (point[0], point[1]), 1, (0, 0, 255), 1)
        return
        
    def draw_obj(self, name_obj: str, i_cam: int):
        i_obj = int(name_obj.split("_")[1]) - 1
        # theta = geometry.rtvec_degree2rad(theta)
        theta = pose_to_rtvec(self.window().objs[i_obj].pose)
        if self.objectName() == "cam_{:d}".format(i_cam + 1):
            self.img_modify = self.img_raw.copy()
            points2d    = self.window().objs[i_obj].views[i_cam].points2d
            indexes3d   = self.window().objs[i_obj].views[i_cam].indexes3d
            points3d    = self.window().objs[i_obj].points3d
            lines       = self.window().objs[i_obj].lines
            camera_pars = self.window().cams[i_cam]
            model       = self.window().objs[i_obj].model

            if self.window().functional_area.rbtn_axis.isChecked():
                Visualizer.draw_axis3d(self.img_modify, camera_pars, rtvec=np.zeros(6), unit_length=1.0, width_line=1) # 世界坐标系
                Visualizer.draw_axis3d(self.img_modify, camera_pars, rtvec=theta,       unit_length=0.1, width_line=2) # 物体坐标系

            if self.window().functional_area.rbtn_model.isChecked() and isinstance(model, List):
                Visualizer.draw_model3d(self.img_modify, model, theta, camera_pars)

            if self.window().functional_area.rbtn_backbone.isChecked() and isinstance(lines, np.ndarray) and isinstance(points2d, np.ndarray):
                Visualizer.draw_backbone2d(self.img_modify, points2d, indexes3d, lines, color=(0, 127, 255), width_line=2)
            if self.window().functional_area.rbtn_backbone.isChecked() and isinstance(lines, np.ndarray) and isinstance(points3d, np.ndarray):
                Visualizer.draw_backbone3d(self.img_modify, points3d, lines, theta, camera_pars, color=(255, 255, 0), width_line=1)

            if self.window().functional_area.rbtn_points2d.isChecked() and isinstance(points2d, np.ndarray):
                #Visualizer.draw_points2d(self.img_modify, points2d, color=(0, 127, 255), radius=3)
                Visualizer.draw_points2d_with_texts(self.img_modify, points2d, indexes3d, color=(0, 127, 255), radius=3)
            if self.window().functional_area.rbtn_points3d.isChecked():
                #Visualizer.draw_points3d(self.img_modify, points3d, theta, camera_pars, color=(255, 255, 0))
                Visualizer.draw_points3d_with_texts(self.img_modify, points3d, theta, camera_pars, color=(255, 255, 0))
        return

    def draw_all(self):
        i_cam  = int(self.objectName().split("_")[1]) - 1
        n_objs = len(self.window().objs)
        for i_obj in range(n_objs):
            name_obj = "obj_{}".format(i_obj + 1)
            self.draw_obj(name_obj, i_cam)
        self.draw_points2d()
        self._update_img()
        return

    def slot_save_image(self):
        self.window().fio.save_image_visualize(mode=self.window().mode, scene=self.window().i_scene, cam=self.objectName(), img=self.img_modify)
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

    # 滚轮事件
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
                self.img_show = self.img_modify.copy()
                self._update_img()
                #self.graphicsView.centerOn(u, v)

            elif evt.angleDelta().y() < 0: # 缩小图片
                self.scale -= self.dscale
                if self.scale < 0.5:
                    self.scale = 0.5
                self.img_show = self.img_modify.copy()
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
                self.draw_all()
                self._update_img()
                print("点 {:d}:\t".format(n_points), point2d)
            return
        elif evt.button() == Qt.RightButton: # 右键双击撤销
            pt_scene = self.graphics_view.mapToScene(evt.x(), evt.y())  #把view坐标转换为场景坐标
            item = self.scene.itemAt(pt_scene,self.graphics_view.transform())  #在场景某点寻找图元--最上面的图元
            if item != None:
                n_points = len(self.points2d_chosen_tmp)
                if n_points > 0:
                    self.points2d_chosen_tmp.pop()
                    self.draw_all()
                    self._update_img()
                    print("删除点 {:d}.".format(n_points))
            return

    def keyPressEvent(self, evt: QtGui.QKeyEvent) -> None:
        """
            键盘事件
        """
        texts_premit = ["{:d}".format(x + 1) for x in list(range((self.n_objs)))]
        if evt.text() in texts_premit:
            i_cam = FileIO.name2index(self.objectName())
            i_obj = int(evt.text()) - 1

            name_obj       = FileIO.index2name("obj", i_obj)
            sub_table_view = self.get_sub_table_view(i_obj)
            n_points_tmp   = len(self.points2d_chosen_tmp)
            n_points_obj   = sub_table_view.array_chosen.shape[0]

            if self.window().dialog_settings.settings.FLAGS_POINTS2D == 0 and self.window().mode=="solve" and i_obj== 0:
                if len(self.points2d_chosen_tmp) >= 5:
                    print("ELL")
                    self.points2d_objs[name_obj] = self.points2d_chosen_tmp
                    self.points2d_chosen_tmp     = []
                    
                    ellipse = Conic.Conic2d()
                    ellipse._set_by_5points2d(np.array(self.points2d_objs[name_obj])[1:])
                    ellipse.draw(self.img_modify)
                    self._update_img()
                    self.window().objs[i_obj].views[i_cam].points2d  = np.array(self.points2d_chosen_tmp)
                    self.window().objs[i_obj].views[i_cam].indexes3d = sub_table_view.indexes_chosen
                    self.window().fio.save_points2d(self.window().mode, self.window().i_scene, i_obj, i_cam, array=self.points2d_chosen_tmp)
                    self.window().fio.save_indexes3d(self.window().mode, self.window().i_scene, i_obj, i_cam, indexes=sub_table_view.indexes_chosen)

            elif n_points_tmp == n_points_obj:
                self.points2d_objs[name_obj] = self.points2d_chosen_tmp
                self.window().objs[i_obj].views[i_cam].points2d  = np.array(self.points2d_chosen_tmp)
                self.window().objs[i_obj].views[i_cam].indexes3d = sub_table_view.indexes_chosen
                self.window().fio.save_points2d(self.window().mode, self.window().i_scene, i_obj, i_cam, array=self.points2d_chosen_tmp)
                self.window().fio.save_indexes3d(self.window().mode, self.window().i_scene, i_obj, i_cam, indexes=sub_table_view.indexes_chosen)

            else:
                print("错误: 点数 {}/{} 不正确. ".format(n_points_tmp, n_points_obj))

            if self.debug:
                print("[DEBUG]:\t<{}>  EMIT SIGNAL <{}>".format(self.objectName(), self.sig_choose_points2d_successed.signal))
            self._update_table_widget_show_points()

        elif evt.text() in [item for item in "qweasdtyughj"]:
            i_obj = self.window().functional_area.tab_widget_objs.currentIndex()
            tmp_pose = self.window().objs[i_obj].pose
            step_r = np.pi / 180
            if evt.text() == "q":
                self.window().objs[i_obj].pose = tmp_pose @ geometry.r_to_R(np.array([ step_r, 0, 0]))
            if evt.text() == "a":
                self.window().objs[i_obj].pose = tmp_pose @ geometry.r_to_R(np.array([-step_r, 0, 0]))
            if evt.text() == "w":
                self.window().objs[i_obj].pose = tmp_pose @ geometry.r_to_R(np.array([0,  step_r, 0]))
            if evt.text() == "s":
                self.window().objs[i_obj].pose = tmp_pose @ geometry.r_to_R(np.array([0, -step_r, 0]))
            if evt.text() == "e":
                self.window().objs[i_obj].pose = tmp_pose @ geometry.r_to_R(np.array([0, 0,  step_r]))
            if evt.text() == "d":
                self.window().objs[i_obj].pose = tmp_pose @ geometry.r_to_R(np.array([0, 0, -step_r]))

            step_t = 0.005
            if evt.text() == "t":
                self.window().objs[i_obj].pose = tmp_pose @ geometry.t_to_T(np.array([ step_t, 0, 0]))
            if evt.text() == "g":
                self.window().objs[i_obj].pose = tmp_pose @ geometry.t_to_T(np.array([-step_t, 0, 0]))
            if evt.text() == "y":
                self.window().objs[i_obj].pose = tmp_pose @ geometry.t_to_T(np.array([0,  step_t, 0]))
            if evt.text() == "h":
                self.window().objs[i_obj].pose = tmp_pose @ geometry.t_to_T(np.array([0, -step_t, 0]))
            if evt.text() == "u":
                self.window().objs[i_obj].pose = tmp_pose @ geometry.t_to_T(np.array([0, 0,  step_t]))
            if evt.text() == "j":
                self.window().objs[i_obj].pose = tmp_pose @ geometry.t_to_T(np.array([0, 0, -step_t]))
            rtvec = geometry.pose_to_rtvec(self.window().objs[i_obj].pose)
            self.window().functional_area.tab_widget_objs.setCurrentIndex(i_obj)
            self.window().functional_area.get_sub_tab_widget(i_obj).set_rtvec(rtvec)
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
