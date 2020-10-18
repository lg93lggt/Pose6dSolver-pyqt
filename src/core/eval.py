import  numpy as np
import  cv2

from . import  geometry as geo
from . import  FileIO 

def to_homo(pts):
    n_pts = pts.shape[0]
    ones = np.ones((n_pts, 1))
    pts_homo = np.hstack((pts, ones))
    return pts_homo

