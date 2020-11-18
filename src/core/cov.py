import numpy as np
import glob
import pandas as pd
import  cv2
from matplotlib import pyplot as plt  
from matplotlib import cm   
from matplotlib import axes  
  
def draw_heatmap(data,xlabels,ylabels):  
    #cmap=cm.Blues      
    cmap=cm.get_cmap('jet',1000)  
    figure=plt.figure(facecolor='w')  
    ax=figure.add_subplot(1,1,1,position=[0.1,0.15,0.8,0.8])  
    ax.set_yticks(range(len(ylabels)))  
    ax.set_yticklabels(ylabels)  
    ax.set_xticks(range(len(xlabels)))  
    ax.set_xticklabels(xlabels)  
    map=ax.imshow(data,interpolation='nearest',cmap=cmap,aspect='auto',vmin=data.min(),vmax=data.max())  
    cb=plt.colorbar(mappable=map,cax=None,ax=None,shrink=0.5)  
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = ax.text(j, i, np.round(data[i, j], 3) ,
                        ha="center", va="center", color="w")

import glob
pths = glob.glob("./姿态测量4/test_cov/*.txt")
pths.sort()

pth = "./姿态测量4/test_cov/cam0_pt0.txt"
img1 = cv2.imread("./姿态测量4/images_solve/cam_1/scene_20.png")
img2 = cv2.imread("./姿态测量4/images_solve/cam_2/scene_20.png")

a_all = []
for pth in pths:
    data = np.loadtxt(pth)
    a_all.append(data[:, 0])
a_all = np.array(a_all)

for pth in pths:
    if "cam0" in pth:
        p2d = np.loadtxt("./姿态测量4/points2d_solve/cam_1/obj_1/scene_20.txt")
        p2d = p2d.astype(np.int)

        data = np.loadtxt(pth)


        a = data[:, 0].reshape((9, 9)).T 
        u = data[0:, 1]
        v = data[0:, 2]

        b = np.zeros((9, 9))
        b = cv2.normalize(a, b, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255
        b = np.round(b)
        b = b.astype(np.uint8)
        c = cv2.applyColorMap(b, cv2.COLORMAP_JET)

        i = pth.find("pt")
        i_pt = int(pth[i+2])
        x = p2d[i_pt]
        img1[x[1]-4 : x[1]+5, x[0]-4:x[0]+5] = c
        cv2.imshow("img1", img1)
        cv2.waitKey(1)
        
    if "cam1" in pth:
        p2d = np.loadtxt("./姿态测量4/points2d_solve/cam_2/obj_1/scene_20.txt")
        p2d = p2d.astype(np.int)

        data = np.loadtxt(pth)


        a = data[:, 0].reshape((9, 9)).T 
        u = data[0:, 1]
        v = data[0:, 2]

        b = np.zeros((9, 9))
        b = cv2.normalize(a, b, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255
        b = np.round(b)
        b = b.astype(np.uint8)
        c = cv2.applyColorMap(b, cv2.COLORMAP_JET)

        i = pth.find("pt") 
        i_pt = int(pth[i+2])
        x = p2d[i_pt]
        img2[x[1]-4 : x[1]+5, x[0]-4:x[0]+5] = c
        cv2.imshow("img2", img2)
        cv2.waitKey(1)
    a_ = (a - a[4, 4]) / a[4, 4] * 100
    draw_heatmap(a_, np.arange(-4, 5, 1), np.arange(-4, 5, 1))
    plt.title("delta_angles(%)")
    plt.xlabel("delta_u")
    plt.ylabel("delta_v")
    plt.savefig(pth + ".jpg") 


cv2.imwrite("./姿态测量4/test_cov/cam0.jpg", img1)
cv2.imwrite("./姿态测量4/test_cov/cam1.jpg", img2)




print()