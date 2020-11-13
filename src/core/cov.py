import numpy as np
import glob
import pandas as pd
from matplotlib import pyplot as plt  
from matplotlib import cm   
from matplotlib import axes  
  
def draw_heatmap(data,xlabels,ylabels):  
    #cmap=cm.Blues      
    cmap=cm.get_cmap('rainbow',1000)  
    figure=plt.figure(facecolor='w')  
    ax=figure.add_subplot(1,1,1,position=[0.1,0.15,0.8,0.8])  
    ax.set_yticks(range(len(ylabels)))  
    ax.set_yticklabels(ylabels)  
    ax.set_xticks(range(len(xlabels)))  
    ax.set_xticklabels(xlabels)  
    map=ax.imshow(data,interpolation='nearest',cmap=cmap,aspect='auto',vmin=a.min(),vmax=a.max())  
    cb=plt.colorbar(mappable=map,cax=None,ax=None,shrink=0.5)  
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = ax.text(j, i, np.round(data[i, j], 3) ,
                        ha="center", va="center", color="w")
    plt.show() 

pth = "/home/veily/桌面/Pose6dSolver-pyqt/姿态测量4/test_cov/1_pt.csv"
data = pd.read_csv(pth)


a = data["angle"].to_numpy().reshape((5, 5)).T 
u = data[" du"].to_numpy()
v = data[" dv"].to_numpy()

draw_heatmap(a, np.arange(2, -3, -1), np.arange(2, -3, -1))

print()