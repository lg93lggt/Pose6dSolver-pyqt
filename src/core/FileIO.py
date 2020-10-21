
import glob
import json
import os
from typing import Any
from easydict import EasyDict
import copy

import cv2
import numpy as np
import shutil   
import argparse 

def copy_file(pth_src: str, pth_dst: str) -> None:
    [dir_file, prefix, suffix] = split_path(pth_dst)
    if not os.path.exists(dir_file):
        make_dir(dir_file)
    shutil.copyfile(pth_src, pth_dst)
    return

def save_numpy_txt(pth: str, array: np.ndarray) -> None:
    [dir_file, _, suffix] = split_path(pth)
    if not os.path.exists(dir_file):
        make_dir(dir_file)
    np.savetxt(pth, array)
    print("\n保存至:\t{}".format(pth))
    return

def savez_numpy_txt(pth: str, array: np.ndarray, indexes: np.ndarray) -> None:
    [dir_file, _, suffix] = split_path(pth)
    if not os.path.exists(dir_file):
        make_dir(dir_file)
    np.savez(pth, array=array, indexes=indexes)
    print("\n保存至:\t{}".format(pth))
    return

def imread(pth_image):
    img = cv2.imdecode(np.fromfile(pth_image, np.uint8), -1)
    return img


def split_path(pth_file: str):
    [dir_file, filename] = os.path.split(pth_file)
    [prefix, suffix] = os.path.splitext(filename)
    return [dir_file, prefix, suffix]

def make_dir(dir_new: str):
    if dir_new == "":
        return
    elif not os.path.exists(dir_new):
        os.makedirs(dir_new)
        print("\n新建文件夹:\t{}".format(dir_new))
        return

def get_sub_dirs_names(dir_motherfolder: str):
    res = os.listdir(dir_motherfolder)
    names_dir_sub = res.copy()
    for [i, item] in enumerate(res):
        suffix = os.path.splitext(item)[1]
        if suffix != "":
            print("\nWARINING: 输入文件夹最好只包含文件夹.\n")
            names_dir_sub.remove(item)
    names_dir_sub.sort()
    return names_dir_sub

def load_camera_pars(pth):
    with open(pth) as fp:
        data = EasyDict(json.load(fp))
        camera_pars= EasyDict({})
        camera_pars.intrin = np.array(data.intrin)
        camera_pars.extrin = np.array(data.extrin)
        camera_pars.rvec = np.array(data.rvec)
        camera_pars.tvec = np.array(data.tvec)
    return camera_pars

def load_model_from_obj(pth_obj: str, is_swap_yz=False):
    """
        加载obj文件
    """
    vertexes = []
    normals = []
    faces = []
    texcoords = []
    norms = []
    material = None

    cnt = 0
    for line in open(pth_obj, "r"):
        cnt+=1
        if line.startswith("#"): 
            continue

        infos = line.split()
        values = []
        for [i_info, info] in enumerate(infos):
            try:
                values.append(float(info))
            except ValueError:
                pass
            
        if not infos: 
            continue
        elif infos[0] == "v":
            v = values 
            if is_swap_yz:
                v = [v[0], v[2], v[1]]
            vertexes.append(v)
        elif infos[0] == "vn":
            vn = values 
            if is_swap_yz:
                vn = [vn[0], vn[2], vn[1]]
            normals.append(vn)
        elif infos[0] == "vt":
            texcoords.append(values)
        elif infos[0] == "f":
            face = []
            for v in infos[1:]:
                w = v.split("//")
                face.append(int(w[0]))
            faces.append(face)
        else:
            pass
    model = {"faces": np.array(faces), "vertexes": np.array(vertexes)}
    return model

def load_model_from_stl_binary(pth_file: str):
        import struct
        
        fp = open(pth_file, "rb")
        h = fp.read(80)

        l = struct.unpack("I", fp.read(4))[0]
        count=0
        model = []
        while True:
            try:
                p = fp.read(12)
                if len(p) == 12:
                    n = struct.unpack("f", p[0:4])[0], struct.unpack("f", p[4:8])[0], struct.unpack("f", p[8:12])[0]
                    
                p = fp.read(12)
                if len(p) == 12:
                    p1 = struct.unpack("f", p[0:4])[0], struct.unpack("f", p[4:8])[0], struct.unpack("f", p[8:12])[0]

                p = fp.read(12)
                if len(p) == 12:
                    p2 = struct.unpack("f", p[0:4])[0], struct.unpack("f", p[4:8])[0], struct.unpack("f", p[8:12])[0]

                p = fp.read(12)
                if len(p) == 12:
                    p3 = struct.unpack("f", p[0:4])[0], struct.unpack("f", p[4:8])[0], struct.unpack("f", p[8:12])[0]
                
                new_tri = (p1, p2, p3)
                model.append(new_tri)
                count += 1
                fp.read(2)

                if len(p)==0:
                    break
            except EOFError:
                break
        fp.close()
        return model



class FileIO(object):
    def __init__(self):
        self.struct = EasyDict({
            "dir_root": "",
            "calib": {
                "n_cams": 0, "n_scenes": 0, "n_objs": 1, "n_models": 1, "unit_length": 0,
                "images": [], "points2d": [], "points3d": [], "models": [], "results": [], "visualize": [], "points3d": [], "logs": []
            },
            "solve": {
                "n_cams": 0, "n_scenes": 0, "n_objs": 0, "n_models": 0, 
                "images": [], "points2d": [], "points3d": [], "models": [], "results": [], "visualize": [], "points3d": [], "logs": []
            }
        })
        self.dir_lv1 = ["images", "points2d", "points3d", "models", "results", "visualize", "points3d", "logs"]
        return

    def new_project(self, project_folder_pth: str="../姿态测量"):
        self.struct.dir_root = os.path.abspath(project_folder_pth)
        make_dir(self.struct.dir_root)
        self.make_dirs()
        self.save_project()
        self.make_cube_points3d_calib()
        return
        
    def make_dirs(self):
        for mode in ["calib", " solve"]:
            for name_dir in self.dir_lv1:
                make_dir(os.path.join(self.struct.dir_root, "{}_{}".format(name_dir, mode)))
        return

    def outprint(self, *args):
        str_out = args[0]
        items = args[1]
        print()
        for item in items:
            print(str_out, "\t", item)
        return

    def make_cube_points3d_calib(self):
        points3d_unit_cube = np.array(
           [[0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1]]
        )
        dir_sub = os.path.join(self.struct.dir_root, "points3d_calib")
        make_dir(dir_sub)
        pth = os.path.join(dir_sub, "obj_1.txt")
        with open(pth, "w") as fp:
            save_numpy_txt(pth, points3d_unit_cube * self.struct.calib.unit_length)
            print(pth)
        return

    def save_project(self):
        pth_project_ini = os.path.join(self.struct.dir_root, "project.ini")
        with open(pth_project_ini, "w", encoding="utf-8") as fp:
            json.dump(self.struct, fp, indent=4, ensure_ascii=False)
            print("更新并保存工程文件.")
        return

    def load_project_from_filedir(self, project_folder_pth: str="./姿态测量"):
        self.struct.dir_root = os.path.abspath(project_folder_pth)
        pth_project_ini = os.path.join(self.struct.dir_root, "project.ini") 
        if not os.path.exists(pth_project_ini):
            raise IOError("未找到 project.ini !")     
        with open(pth_project_ini, encoding="utf-8") as fp:
            self.struct = EasyDict(json.loads(fp.read()))
        return self.struct

    def _update(self):
        self.match_pairs("calib")
        self.match_pairs("solve")
        self.save_project()
        self.make_cube_points3d_calib()
        return

    def set_unit_length(self, length: Any):
        try:
            self.struct.calib.unit_length = float(length)
        except ValueError:
            print("字符串格式错误.")
        return

    def match_pairs(self, mode: str):
        n_scenes = self.struct[mode].n_scenes
        n_cams   = self.struct[mode].n_cams
        n_objs   = self.struct[mode].n_objs if (self.struct[mode].n_objs > 0) else 1
        
        #if (n_cams > 0) and (n_scenes > 0):
        pairs_scene = []
        for i_scene in range(n_scenes):
            pairs_cams = []
            for i_cam in range(n_cams):
                pairs_cams.append(os.path.join("images_" + mode, self.index2name("cam", i_cam), self.index2name("scene", i_scene) + ".png"))
            pairs_scene.append(pairs_cams)
        self.struct[mode].images = pairs_scene

        #------------------------------------------------------------#
        list2d = []
        list3d = []
        for i_scene in range(n_scenes):
            name_scene = self.index2name("scene", i_scene)
            list_scense2d = []
            list_scense3d = []
            for i_obj in range(n_objs):
                list_obj_2d = []
                list_obj_3d = []
                name_obj = self.index2name("obj", i_obj)
                pth_points3d = os.path.join("points3d_" + mode, name_obj + ".txt")
                for i_cam in range(n_cams):
                    name_cam = self.index2name("cam", i_cam)
                    pth_points2d = os.path.join("points2d_" + mode, name_cam, name_obj, name_scene + ".txt")
                    pth_points3d = os.path.join("points3d_" + mode, name_cam, name_obj, name_scene + ".npz")
                    list_obj_2d.append(pth_points2d)
                    list_obj_3d.append(pth_points3d)
                list_scense2d.append(list_obj_2d)
                list_scense3d.append(list_obj_3d)
            list2d.append(list_scense2d)
            list3d.append(list_scense3d)
            self.struct[mode].points2d = list2d.copy()
            self.struct[mode].points3d = list3d.copy()
        self.save_project()
        return

    def load_images_from_motherfolder_dir(self, dir_motherfolder: str, mode: str=""):
        """
            加载图像文件夹
        """
        names_dir_sub = get_sub_dirs_names(dir_motherfolder)
        pths_input_images = []
        suffixes_obj = [".bmp", ".jpg", ".png"] # 可选图片后缀
        for name_sub_dir in names_dir_sub: # n个相机文件夹
            dir_sub = os.path.join(dir_motherfolder, name_sub_dir)
            for suffix_obj in suffixes_obj:
                pth_obj = os.path.join(dir_sub, "*" + suffix_obj)
                tmp = glob.glob(pth_obj)
                if tmp != []:
                    tmp.sort()
                    pths_input_images.append(tmp) 
        n_cams   = len(pths_input_images)
        n_scenes = len(pths_input_images[0])

        if not (mode in self.struct.keys()):
            return pths_input_images

        self.struct[mode].n_cams   = n_cams
        self.struct[mode].n_scenes = n_scenes

        if self.struct[mode].n_cams > 0:
            for i_cam in range(self.struct[mode].n_cams): # 复制标定图像至标定图像文件夹
                dir_new = os.path.join(self.struct.dir_root, "images_" + mode, self.index2name("cam", i_cam))
                for i_scene in range(self.struct[mode].n_scenes):
                    name_scene = "scene_{:d}".format(i_scene + 1)
                    pth_old = pths_input_images[i_cam][i_scene]
                    pth_new = os.path.join(self.struct.dir_root, dir_new, name_scene + ".png")
                    copy_file(pth_old, pth_new)
                    print("复制:\t{}\t到:\{}".format(pth_old, pth_new))
        self.match_pairs(mode)
        self.save_project()
        return pths_input_images

    def load_points3d_from_motherfolder_dir(self, dir_motherfolder: str, mode: str=""):
        """
            加载3d点文件夹
        """
        suffixes_obj      = [".txt"] # 可选后缀
        pths_input_models = []
        for suffix_obj in suffixes_obj:
            pth_obj = os.path.join(dir_motherfolder, "*" + suffix_obj)
            pths_input_models = glob.glob(pth_obj)
            if pths_input_models != []:
                pths_input_models.sort()
        n_objs = len(pths_input_models)
        self.struct[mode].n_objs = n_objs
        if not (mode in self.struct.keys()):
            return pths_input_models

        self.make_dirs()
        #self.make_file_structure_subdirs(mode)

        self.outprint("{}:".format("load"), pths_input_models)
        if n_objs > 0:
            for i_obj in range(n_objs): # 复制至文件夹
                pth_old = pths_input_models[i_obj]
                name_obj = "obj_{:d}".format(i_obj + 1)
                pth_new = os.path.join(self.struct.dir_root, "points3d_" + mode, name_obj + ".txt")
                copy_file(pth_old, pth_new)
                print("复制:\t{}\t到:\{}".format(pth_old, pth_new))
        self.save_project()
        return pths_input_models

    def load_modeles_from_motherfolder_dir(self, dir_motherfolder: str, mode: str=""):
        """
            加载模型文件夹
        """
        suffixes_obj      = [".stl"] # 可选后缀
        pths_input_models = []
        for suffix_obj in suffixes_obj:
            pth_obj = os.path.join(dir_motherfolder, "*" + suffix_obj)
            pths_input_models = glob.glob(pth_obj)
            if pths_input_models != []:
                pths_input_models.sort()
        n_models = len(pths_input_models)
        self.struct[mode].n_models = n_models
        
        if not (mode in self.struct.keys()):
            return pths_input_models

        self.make_dirs()
        #self.make_file_structure_subdirs(mode)

        self.outprint("{}:".format("load"), pths_input_models)
        if n_models > 0:
            for i_obj in range(n_models): # 复制至文件夹
                pth_old = pths_input_models[i_obj]
                name_model = "obj_{:d}".format(i_obj + 1)
                pth_new = os.path.join(self.struct.dir_root, "models_" + mode, name_model + ".stl")
                copy_file(pth_old, pth_new)
                print("复制:\t{}\t到:\{}".format(pth_old, pth_new))
        self.save_project()
        return pths_input_models

    def name2index(self, name: str):
        return int(name.split("_")[1]) - 1

    def index2name(self, name: str, index: int):
        return "{}_{:d}".format(name, index + 1)

    def load_points2d(self, mode: str, scene: str or int, obj: str or int, cam: str or int):
        if isinstance(scene, str):
            scene = self.name2index(scene)
        if isinstance(obj, str): 
            obj = self.name2index(obj)
        if isinstance(cam, str): 
            cam = self.name2index(cam)
        pth = os.path.join(
            self.struct.dir_root,
            self.struct[mode].points2d[scene][obj][cam]
        )
        if not os.path.exists(pth):
            print("文件不存在", pth)
            return 
        else:
            print("加载:", pth)
            poins2d = np.loadtxt(pth)
            return poins2d.astype(np.int)

    def load_points3d(self, mode: str, obj: str or int):
        if isinstance(obj, str): 
            obj = self.name2index(obj)
        pth = os.path.join(
            self.struct.dir_root,
            "points3d_" + mode,
            (self.index2name("obj", obj) if isinstance(obj, int) else obj) + ".txt"
        )
        if not os.path.exists(pth):
            print("文件不存在", pth)
            return 
        else:
            print("加载:", pth)
            return  np.loadtxt(pth)

    def loadz_points3d(self, mode: str, scene: str or int, obj: str or int, cam: str or int):
        if isinstance(scene, str):
            scene = self.name2index(scene)
        if isinstance(obj, str): 
            obj = self.name2index(obj)
        if isinstance(cam, str): 
            cam = self.name2index(cam)
        pth = os.path.join(
            self.struct.dir_root,
            self.struct[mode].points3d[scene][obj][cam]
        )
        if not os.path.exists(pth):
            print("文件不存在", pth)
            return 
        else:
            print("加载:", pth)
            points3dz = np.load(pth)
            return points3dz

    def save_points2d(self, mode: str, scene: str or int, obj: str or int, cam: str or int, array: np.ndarray) -> None:
        if isinstance(scene, str):
            scene = self.name2index(scene)
        if isinstance(obj, str): 
            obj = self.name2index(obj)
        if isinstance(cam, str): 
            cam = self.name2index(cam)
        pth = os.path.join(self.struct.dir_root, self.struct[mode].points2d[scene][obj][cam])
        save_numpy_txt(pth, array)
        print("\n保存至:\t{}".format(pth))
        return

    def save_points3d(self, mode: str, scene: str or int, obj: str or int, cam: str or int, array: np.ndarray) -> None:
        pth = os.path.join(self.struct.dir_root, self.struct[mode].points3d[scene][obj][cam])
        save_numpy_txt(pth, array)
        print("\n保存至:\t{}".format(pth))
        return

    def savez_points3d(self, mode: str, scene: str or int, obj: str or int, cam: str or int,array: np.ndarray, indexes: np.ndarray) -> None:
        if isinstance(scene, str):
            scene = self.name2index(scene)
        if isinstance(obj, str): 
            obj = self.name2index(obj)
        if isinstance(cam, str): 
            cam = self.name2index(cam)
        pth = os.path.join(self.struct.dir_root, self.struct[mode].points3d[scene][obj][cam])
        savez_numpy_txt(pth, array, indexes)
        print("\n保存至:\t{}".format(pth))
        return

    def load_image_raw(self, mode: str, scene: str or int, cam: str or int):
        if isinstance(scene, str):
            scene = self.name2index(scene)
        if isinstance(cam, str): 
            cam = self.name2index(cam)
        pth = os.path.join(self.struct.dir_root, self.struct[mode].images[scene][cam])
        if os.path.exists(pth):
            print("加载:", pth)
            return imread(pth)
        else:
            return

    def load_image_visualize(self, mode: str, scene: str or int, obj: str or int, cam: str or int):
        if isinstance(scene, str):
            scene = self.name2index(scene)
        if isinstance(obj, str): 
            obj = self.name2index(obj)
        if isinstance(cam, str): 
            cam = self.name2index(cam)
        pth = os.path.join(
            self.struct.dir_root,
            self.struct[mode].visualize[scene][obj][cam]
        )
        return imread(pth)

    def save_camera_pars(self, cam: int or str, camera_pars):
        if isinstance(cam, int):
            namse_cam = self.index2name("cam", cam)
        else:
            namse_cam = cam
        dir_ = os.path.join(
            self.struct.dir_root, 
            "results_calib",
            namse_cam)
        make_dir(dir_)
        pth = os.path.join(dir_, "camera_pars.json")
        camera_pars = EasyDict(camera_pars)
        with open(pth, "w") as fp:
            dict_ouput = EasyDict({})
            dict_ouput.intrin = camera_pars.intrin.tolist()
            dict_ouput.extrin = camera_pars.extrin.tolist()
            dict_ouput.rvec = camera_pars.rvec.tolist()
            dict_ouput.tvec = camera_pars.tvec.tolist()
            json.dump(dict_ouput, fp, indent=4)
            print("保存: ", pth)
        return

    def load_camera_pars(self, cam: str or int):
        if isinstance(cam, int): 
            cam = self.index2name("cam", cam)
        pth = os.path.join(
            self.struct.dir_root, 
            "results_calib",
            cam,
            "camera_pars.json"
        )
        if os.path.exists(pth):
            with open(pth) as fp:
                data = EasyDict(json.load(fp))
                camera_pars= EasyDict({})
                camera_pars.intrin = np.array(data.intrin)
                camera_pars.extrin = np.array(data.extrin)
                camera_pars.rvec = np.array(data.rvec)
                camera_pars.tvec = np.array(data.tvec)
            print("加载:", pth)
            return camera_pars
        else:
            print(cam, "不存在", pth)
            return

    def save_image_visualize(self, mode: str, name_scene: str, name_obj: str, name_cam: str, img: np.ndarray):
        pth = os.path.join(
            self.struct.dir_root, 
            self.struct[mode].dirs.visualize,
            self.struct[mode].pairs[name_scene][self.name2index(name_obj)][self.name2index(name_cam)])
        cv2.imwrite(pth, img)
        print("保存图像: ", pth)
        return

    def save_chosen_points3d(self, mode: str, scene: str, obj: str, cam: str, points3d: np.ndarray):
        if isinstance(scene, str):
            scene = self.name2index(scene)
        if isinstance(obj, str): 
            obj = self.name2index(obj)
        if isinstance(cam, str): 
            cam = self.name2index(cam)
        pth = os.path.join(
            self.struct.dir_root,
            self.struct[mode].points3d[scene][obj][cam]
        )
        save_numpy_txt(pth, points3d)
        print("保存: ", pth)
        return

    def load_log_from_file(self, mode: str, scene: str, obj: str, cam: str, points3d: np.ndarray):
        if isinstance(scene, str):
            scene = self.name2index(scene)
        if isinstance(obj, str): 
            obj = self.name2index(obj)
        if isinstance(cam, str): 
            cam = self.name2index(cam)
        pth = os.path.join(
            self.struct.dir_root,
            self.struct[mode].logs[scene][obj][cam]
        )
        if os.path.exists(pth):
            print("加载:", pth)
            return np.loadtxt(pth)
        else:
            return    

    def save_log(self, mode: str, scene: str or int, obj: str or int, log: np.ndarray):
        if isinstance(scene, int):
            scene = self.index2name("scene", scene)
        if isinstance(obj, int):
            obj = self.index2name("obj", obj)
        pth = os.path.join(
            self.struct.dir_root, 
            "logs_" + mode,
            obj,
            scene + ".txt"
        )
        save_numpy_txt(pth, log)
        print("记录保存:\t", pth)
        return

    def save_theta(self, mode: str,  scene: str or int, obj: str or int,  theta: np.ndarray):
        if isinstance(scene, int):
            scene = self.index2name("scene", scene)
        if isinstance(obj, int):
            obj = self.index2name("obj", obj)
        pth = os.path.join(
            self.struct.dir_root, 
            "results_" + mode,
            obj,
            scene + ".txt"
        )
        save_numpy_txt(pth, theta)
        print("姿态保存:\t", pth)
        return

    def load_model(self, mode: str, obj: str or int):
        if isinstance(obj, str): 
            obj = self.name2index(obj)
        pth = os.path.join(
            self.struct.dir_root,
            "models_" + mode,
            (self.index2name("obj", obj) if isinstance(obj, int) else obj) + ".stl"
        )
        if not os.path.exists(pth):
            print("文件不存在", pth)
            return 
        else:
            print("加载:", pth)
            return  load_model_from_stl_binary(pth)



def t():
    fio = FileIO()
    dir_project ="./测试"
    if os.path.exists(dir_project):
        fio.load_project_from_filedir(dir_project)
    else:
        fio.new_project(dir_project)
    fio.load_images_from_motherfolder_dir("C:/Users/Li/Desktop/Pose6dSolver-pyqt/图像/CamCalib", "calib")
    fio.load_modeles_from_motherfolder_dir("C:/Users/Li/Desktop/Pose6dSolver-pyqt/图像/model/2", "calib")
    fio.load_images_from_motherfolder_dir("C:/Users/Li/Desktop/Pose6dSolver-pyqt/测试/images_solve", "solve")
    fio.load_modeles_from_motherfolder_dir("C:/Users/Li/Desktop/Pose6dSolver-pyqt/图像/model/1", "solve")
    fio.struct.solve.n_objs = 2
    fio.match_pairs("solve")
    fio.match_pairs("calib")
    return  fio

def main(args):
    fio = FileIO()
    dir_project = args.project
    dir_input_images_calib =  args.load_calib
    dir_input_images_solve =  args.load_solve
    if os.path.exists(dir_project):
        fio.load_project_from_file(dir_project)
    else:
        fio.new_project(dir_project)
        if dir_input_images_calib:
            fio.load_images_from_motherfolder_dir(dir_input_images_calib, mode="calib")
        if dir_input_images_solve:
            fio.load_images_from_motherfolder_dir(dir_input_images_solve, mode="solve")
    return  fio
    


if __name__ == "__main__":
    t()
    parser = argparse.ArgumentParser("11")
    parser.add_argument(
        "--project",
        nargs='?',
        type=str,
        const="./姿态测量",
        required=False,
        help="加载工程, 若不存在则新建工程."
    )
    parser.add_argument(
        "-load_calib",
        type=str,
        required=False,
        help="标定图像文件夹."
    )
    parser.add_argument(
        "-load_solve",
        type=str,
        required=False,
        help="测量图像文件夹."
    )
    args = parser.parse_args()
    main(args)
    print()    
