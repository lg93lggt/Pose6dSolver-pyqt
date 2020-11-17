
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

def save_numpy_txt(pth: str, array: np.ndarray, format="%.4f") -> None:
    [dir_file, _, suffix] = split_path(pth)
    if not os.path.exists(dir_file):
        make_dir(dir_file)
    np.savetxt(pth, array, format)
    print("保存至:\t{}".format(pth))
    return
    
def load_numpy_txt(pth: str, dtype=float) -> np.ndarray:
    if not os.path.exists(pth):
        print("错误: 文件不存在.")
        return
    else:
        print("打开:\t{}".format(pth))
        return np.loadtxt(pth, dtype=dtype)

def savez_numpy_txt(pth: str, array: np.ndarray, indexes: np.ndarray) -> None:
    [dir_file, _, suffix] = split_path(pth)
    if not os.path.exists(dir_file):
        make_dir(dir_file)
    np.savez(pth, array=array, indexes=indexes)
    print("\n保存至:\t{}".format(pth))
    return

def imread(pth_image: str):
    img = cv2.imdecode(np.fromfile(pth_image, np.uint8), -1)
    return img

def imwrite(pth_image: str, image: np.ndarray):
    [dir_image, prefix_image, suffix_image] = split_path(pth_image)
    make_dir(dir_image)
    cv2.imencode("." + suffix_image, image)[1].tofile(pth_image)
    print("保存图像: ", pth_image)
    return

def split_path(pth_file: str):
    [dir_file, filename] = os.path.split(pth_file)
    [prefix, suffix] = os.path.splitext(filename)
    return [dir_file, prefix, suffix]

def make_dir(dir_new: str):
    if dir_new == "":
        return
    elif not os.path.exists(dir_new):
        os.makedirs(dir_new)
        print("新建文件夹:\t{}".format(dir_new))
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

def name2index(name: str):
    return int(name.split("_")[1]) - 1

def index2name(name: str, index: int):
    return "{}_{:d}".format(name, index + 1)


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
        self.make_cube_calib()
        return
        
    def make_dirs(self):
        for mode in ["calib", "solve"]:
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

    def make_cube_calib(self):
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
        points3d_cube = points3d_unit_cube * self.struct.calib.unit_length
        dir_sub = os.path.join(self.struct.dir_root, "points3d_calib")
        make_dir(dir_sub)
        pth = os.path.join(dir_sub, "obj_1.txt")
        with open(pth, "w") as fp:
            print("\n生成标定架3D关键点:")
            save_numpy_txt(pth, points3d_cube)

        lines_cube = np.array([
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ], dtype=np.int)
        pth = os.path.join(dir_sub, "lines_1.txt")
        with open(pth, "w") as fp:
            print("\n生成标定架骨架:")
            save_numpy_txt(pth, lines_cube, format="%d")
        return

    def save_project(self):
        pth_project_ini = os.path.join(self.struct.dir_root, "project.ini")
        with open(pth_project_ini, "w", encoding="utf-8") as fp:
            json.dump(self.struct, fp, indent=4, ensure_ascii=False)
            print("\n保存工程配置文件.")
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
        self.make_cube_calib()
        return

    def set_unit_length(self, length: Any):
        try:
            self.struct.calib.unit_length = float(length)
        except ValueError:
            print("错误: 字符串格式错误.")
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
                pairs_cams.append(os.path.join("images_" + mode, index2name("cam", i_cam), index2name("scene", i_scene) + ".png"))
            pairs_scene.append(pairs_cams)
        self.struct[mode].images = pairs_scene

        #------------------------------------------------------------#
        list2d = []
        list3d = []
        for i_scene in range(n_scenes):
            name_scene = index2name("scene", i_scene)
            list_scense2d = []
            list_scense3d = []
            for i_obj in range(n_objs):
                list_obj_2d = []
                list_obj_3d = []
                name_obj = index2name("obj", i_obj)
                pth_indexes3d = os.path.join("points3d_" + mode, name_obj + ".txt")
                for i_cam in range(n_cams):
                    name_cam = index2name("cam", i_cam)
                    pth_points2d = os.path.join("points2d_" + mode, name_cam, name_obj, name_scene + ".txt")
                    pth_indexes3d = os.path.join("points3d_" + mode, name_cam, name_obj, name_scene + ".txt")
                    list_obj_2d.append(pth_points2d)
                    list_obj_3d.append(pth_indexes3d)
                list_scense2d.append(list_obj_2d)
                list_scense3d.append(list_obj_3d)
            list2d.append(list_scense2d)
            list3d.append(list_scense3d)
            self.struct[mode].points2d = list2d.copy()
            self.struct[mode].points3d = list3d.copy()
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
        if pths_input_images == []:
            print("\n错误: 未找到图像.")
            return
        
        n_cams   = len(pths_input_images)
        n_scenes = len(pths_input_images[0])

        if not (mode in self.struct.keys()):
            return pths_input_images

        self.struct[mode].n_cams   = n_cams
        self.struct[mode].n_scenes = n_scenes

        if self.struct[mode].n_cams > 0:
            for i_cam in range(self.struct[mode].n_cams): # 复制标定图像至标定图像文件夹
                dir_new = os.path.join(self.struct.dir_root, "images_" + mode, index2name("cam", i_cam))
                for i_scene in range(self.struct[mode].n_scenes):
                    name_scene = "scene_{:d}".format(i_scene + 1)
                    pth_old = pths_input_images[i_cam][i_scene]
                    pth_new = os.path.join(self.struct.dir_root, dir_new, name_scene + ".png")
                    copy_file(pth_old, pth_new)
                    print("复制:\t{}\t到:\{}".format(pth_old, pth_new))
        self.match_pairs(mode)
        return 

    def load_points3d_from_motherfolder_dir(self, dir_motherfolder: str, mode: str=""):
        """
            加载3d点文件夹
        """
        suffixes_obj      = [".txt"] # 可选后缀
        pths_input_points3d = []
        for suffix_obj in suffixes_obj:
            pth_obj = os.path.join(dir_motherfolder, "*" + suffix_obj)
            pths_input_points3d = glob.glob(pth_obj)
            if pths_input_points3d != []:
                pths_input_points3d.sort()
        n_objs = len(pths_input_points3d)
        self.struct[mode].n_objs = n_objs
        if not (mode in self.struct.keys()):
            return pths_input_points3d

        self.make_dirs()
        #self.make_file_structure_subdirs(mode)

        if n_objs > 0:
            for i_obj in range(n_objs): # 复制至文件夹
                pth_old = pths_input_points3d[i_obj]
                name_obj = index2name("obj", i_obj)
                pth_new = os.path.join(self.struct.dir_root, "points3d_" + mode, name_obj + ".txt")
                copy_file(pth_old, pth_new)
                print("复制:\t{}\t到:\{}".format(pth_old, pth_new))
        return pths_input_points3d

    def load_lines_from_motherfolder_dir(self, dir_motherfolder: str, mode: str=""):
        """
            加载骨架连线文件夹
        """
        suffixes_obj      = [".txt"] # 可选后缀
        pths_input_lines = []
        for suffix_obj in suffixes_obj:
            pth_obj = os.path.join(dir_motherfolder, "*" + suffix_obj)
            pths_input_lines = glob.glob(pth_obj)
            if pths_input_lines != []:
                pths_input_lines.sort()
        n_objs = len(pths_input_lines)
        self.struct[mode].n_objs = n_objs
        if not (mode in self.struct.keys()):
            return pths_input_lines

        self.make_dirs()
        #self.make_file_structure_subdirs(mode)

        if n_objs > 0:
            for i_obj in range(n_objs): # 复制至文件夹
                pth_old = pths_input_lines[i_obj]
                name_obj = index2name("lines", i_obj)
                pth_new = os.path.join(self.struct.dir_root, "points3d_" + mode, name_obj + ".txt")
                copy_file(pth_old, pth_new)
                print("复制:\t{}\t到:\{}".format(pth_old, pth_new))
        return pths_input_lines

    def load_modeles_from_motherfolder_dir(self, dir_motherfolder: str, mode: str=""):
        """
            加载模型文件夹
        """
        suffixes_obj      = [".stl", ".STL"] # 可选后缀
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
                name_model = index2name("obj", i_obj)
                pth_new = os.path.join(self.struct.dir_root, "models_" + mode, name_model + ".stl")
                copy_file(pth_old, pth_new)
                print("复制:\t{}\t到:\{}".format(pth_old, pth_new))
        self.save_project()
        return pths_input_models

    def load_points2d(self, mode: str, scene: str or int, obj: str or int, cam: str or int):
        [ret, data]  = [False, None]
        if isinstance(scene, str):
            scene = name2index(scene)
        if isinstance(obj, str): 
            obj = name2index(obj)
        if isinstance(cam, str): 
            cam = name2index(cam)
        pth = os.path.join(
            self.struct.dir_root,
            self.struct[mode].points2d[scene][obj][cam]
        )
        print("加载 图像{}/物体{}/相机{} 2D关键点:".format(scene + 1, obj + 1, cam + 1))
        if not os.path.exists(pth):
            print("错误: 文件不存在.\n")
        else:
            poins2d = load_numpy_txt(pth)
            ret     = True
            data    = poins2d.astype(np.int)
        return [ret, data]

    def load_points3d(self, mode: str, obj: str or int):
        [ret, data]  = [False, None]
        if isinstance(obj, int): 
            obj = index2name("obj", obj)
        pth = os.path.join(
                self.struct.dir_root,
                "points3d_" + mode,
                obj  + ".txt"
            )
        if not os.path.exists(pth):
            print("错误: 文件不存在.\n")
             
        else:
            print("加载 物体{} 3D关键点:".format(name2index(obj) + 1))
            ret  = True
            data = load_numpy_txt(pth)
        return [ret, data]

    def load_indexes3d(self, mode: str, scene: str or int, obj: str or int, cam: str or int):
        [ret, data]  = [False, None]
        if isinstance(scene, str):
            scene = name2index(scene)
        if isinstance(obj, str): 
            obj = name2index(obj)
        if isinstance(cam, str): 
            cam = name2index(cam)
        pth = os.path.join(
            self.struct.dir_root,
            self.struct[mode].points3d[scene][obj][cam]
        )
        print("加载 图像{}/物体{}/相机{} 3D关键点索引:".format(scene + 1, obj + 1, cam + 1))
        if not os.path.exists(pth):
            print("错误: 文件不存在.\n")
        else:
            ret  = True
            data = load_numpy_txt(pth, dtype=np.int)
        return [ret, data]

    def load_backbonelines(self, mode: str, obj: str or int):
        [ret, data]  = [False, None]
        if isinstance(obj, str): 
            obj = name2index(obj)
        pth = os.path.join(
            self.struct.dir_root,
            "points3d_" + mode,
            (index2name("lines", obj) if isinstance(obj, int) else obj) + ".txt"
        )
        if not os.path.exists(pth):
            print("错误: 文件不存在.\n")
             
        else:
            print("加载 物体{} 3D关键点连线:".format(obj + 1))
            ret  = True
            data = load_numpy_txt(pth)
            data = data.astype(np.int)
        return [ret, data]

    def load_image_raw(self, mode: str, scene: str or int, cam: str or int):
        [ret, data]  = [False, None]
        if isinstance(scene, str):
            scene = name2index(scene)
        if isinstance(cam, str): 
            cam = name2index(cam)
        pth = os.path.join(self.struct.dir_root, self.struct[mode].images[scene][cam])
        if os.path.exists(pth):
            print("加载图像:", pth)
            ret  = True
            data = imread(pth)
        else:
            print("错误: 文件不存在.\n")
        return [ret, data]

    def load_image_visualize(self, mode: str, scene: str or int, obj: str or int, cam: str or int):
        [ret, data]  = [False, None]
        if isinstance(scene, str):
            scene = name2index(scene)
        if isinstance(obj, str): 
            obj = name2index(obj)
        if isinstance(cam, str): 
            cam = name2index(cam)
        pth = os.path.join(
            self.struct.dir_root,
            self.struct[mode].visualize[scene][obj][cam]
        )
        if os.path.exists(pth):
            data = imread(pth)
            ret  = True
        return [ret, data]

    def load_camera_pars(self, cam: str or int):
        [ret, data]  = [False, None]
        if isinstance(cam, int): 
            cam = index2name("cam", cam)
        pth = os.path.join(
            self.struct.dir_root, 
            "results_calib",
            cam,
            "camera_pars.json"
        )
        print("加载 相机{} 参数:".format(name2index(cam) + 1))
        if os.path.exists(pth):
            with open(pth) as fp:
                data = EasyDict(json.load(fp))
                camera_pars= EasyDict({})
                camera_pars.intrin = np.array(data.intrin)
                camera_pars.extrin = np.array(data.extrin)
                camera_pars.rvec = np.array(data.rvec)
                camera_pars.tvec = np.array(data.tvec)
            ret  = True
            data = camera_pars
            print("打开:\t", pth)
        else:
            print("错误: 相机未标定.\n")
        return [ret, data]

    def load_log(self, mode: str, scene: str, obj: str):
        if isinstance(scene, int):
            scene = index2name("scene", scene)
        if isinstance(obj, int): 
            obj = index2name("obj", obj)
        pth = os.path.join(
            self.struct.dir_root,
            "logs_" + mode,
            obj,
            scene+".txt"
        )
        if os.path.exists(pth):
            print("加载:", pth)
            return np.loadtxt(pth)
        else:
            return    

    def load_theta(self, scene: str or int, obj: str or int):
        [ret, data]  = [False, None]
        if isinstance(scene, int):
            scene = index2name("scene", scene)
        if isinstance(obj, int):
            obj = index2name("obj", obj)
        pth = os.path.join(
            self.struct.dir_root, 
            "results_solve",
            obj,
            scene + ".txt"
        )
        print("加载 物体{}/图像{} 姿态:".format(name2index(obj) + 1, name2index(scene) + 1))
        data = load_numpy_txt(pth)
        if not(data is None):
            ret = True
        return [ret, data]

    def load_model(self, mode: str, obj: str or int):
        [ret, data]  = [False, None]
        if isinstance(obj, str): 
            obj = name2index(obj)
        pth = os.path.join(
            self.struct.dir_root,
            "models_" + mode,
            (index2name("obj", obj) if isinstance(obj, int) else obj) + ".stl"
        )
        print("加载 物体{} 模型:".format(obj + 1))
        if not os.path.exists(pth):
            print("错误: 文件不存在.\n")
        else:
            data = load_model_from_stl_binary(pth)
            ret  = True
        return [ret, data]

    def save_points2d(self, mode: str, scene: str or int, obj: str or int, cam: str or int, array: np.ndarray) -> None:
        if isinstance(scene, str):
            scene = name2index(scene)
        if isinstance(obj, str): 
            obj = name2index(obj)
        if isinstance(cam, str): 
            cam = name2index(cam)
        pth = os.path.join(self.struct.dir_root, self.struct[mode].points2d[scene][obj][cam])
        print("保存 图像{}/物体{}/相机{} 2D关键点:".format(scene + 1, obj + 1, cam + 1))
        save_numpy_txt(pth, array)
        return

    def save_points3d(self, mode: str, scene: str or int, obj: str or int, cam: str or int, array: np.ndarray) -> None:
        pth = os.path.join(self.struct.dir_root, self.struct[mode].indexes3d[scene][obj][cam])
        print("\n保存3D关键点:")
        save_numpy_txt(pth, array)
        return

    def save_indexes3d(self, mode: str, scene: str or int, obj: str or int, cam: str or int, indexes: np.ndarray) -> None:
        if isinstance(scene, str):
            scene = name2index(scene)
        if isinstance(obj, str): 
            obj = name2index(obj)
        if isinstance(cam, str): 
            cam = name2index(cam)
        pth = os.path.join(
            self.struct.dir_root, 
            self.struct[mode].points3d[scene][obj][cam])
        print("保存 图像{}/物体{}/相机{} 3D关键点索引:".format(scene + 1, obj + 1, cam + 1))
        save_numpy_txt(pth, indexes, format="%d")
        return

    def save_camera_pars(self, cam: int or str, camera_pars):
        if isinstance(cam, int):
            namse_cam = index2name("cam", cam)
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

    def save_image_visualize(self, mode: str, scene: str or int, cam: str or int, img: np.ndarray):
        if isinstance(scene, int):
            scene = index2name("scene", scene)
        if isinstance(cam, int): 
            cam = index2name("cam", cam)
        pth = os.path.join(
                self.struct.dir_root, 
                "visualize_{}".format(mode),
                cam,
                scene + ".jpg"
            )
        print("\n保存图像: ", pth)
        imwrite(pth, img)
        return

    def save_chosen_points3d(self, mode: str, scene: str, obj: str, cam: str, points3d: np.ndarray):
        if isinstance(scene, str):
            scene = name2index(scene)
        if isinstance(obj, str): 
            obj = name2index(obj)
        if isinstance(cam, str): 
            cam = name2index(cam)
        pth = os.path.join(
            self.struct.dir_root,
            self.struct[mode].points3d[scene][obj][cam]
        )
        save_numpy_txt(pth, points3d)
        print("保存: ", pth)
        return

    def save_log(self, mode: str, scene: str or int, obj: str or int, log: np.ndarray):
        if isinstance(scene, int):
            scene = index2name("scene", scene)
        if isinstance(obj, int):
            obj = index2name("obj", obj)
        pth = os.path.join(
            self.struct.dir_root, 
            "logs_" + mode,
            obj,
            scene + ".txt"
        )
        save_numpy_txt(pth, log)
        print("记录保存:\t", pth)
        return

    def save_theta(self, scene: str or int, obj: str or int, theta: np.ndarray):
        if isinstance(scene, int):
            scene = index2name("scene", scene)
        if isinstance(obj, int):
            obj = index2name("obj", obj)
        pth = os.path.join(
            self.struct.dir_root, 
            "results_solve",
            obj,
            scene + ".txt"
        )
        save_numpy_txt(pth, theta, format="%.6f")
        print("姿态保存:\t", pth)
        return

    def update_mode(self, mode: str):
        if   mode == "calib":
            tmp_str = "标定"
        elif mode == "solve":
            tmp_str = "测量"
        else:
            raise TypeError(mode, ": 错误的模式.")
        
        print("\n加载已有{}数据:".format(tmp_str))
        objs = []
        for i_obj in range(self.struct[mode].n_objs):
            obj = EasyDict({})
            [ret, pts3d] = self.load_points3d(mode, i_obj)
            obj.points3d = pts3d

            [ret, lines] = self.load_backbonelines(mode, i_obj)
            obj.lines = lines

            [ret, model] = self.load_model(mode, i_obj)
            obj.model = model

            obj.pose = np.eye(4)

            obj.views = []
            for i_cam in range(self.struct[mode].n_cams):
                view = EasyDict({})
                view.points2d  = None
                view.indexes3d = None
                obj.views.append(view)
            objs.append(obj)

        cams = []
        for i_cam in range(self.struct[mode].n_cams):
            [ret, cam_pars] = self.load_camera_pars(i_cam)
            if ret:
                cams.append(cam_pars)
            else:
                cams.append(None)
        return [objs, cams]



if __name__ == "__main__":
    fio = FileIO()
    fio.load_project_from_filedir("../../姿态测量")
    res1 = fio.update_mode("solve")
    res2 = fio.update_mode("calib")
    res2 = fio.update_mode("init")
    print()

