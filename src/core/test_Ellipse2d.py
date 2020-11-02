
import sys
import numpy as np
sys.path.append("..")
from core import Ellipse2d
from  math import sin, cos, sqrt, atan2
import matplotlib.pyplot as plt

import sympy




def get_ellipse(e_x, e_y, a, b, e_angle):
    angles_circle = np.arange(0, 2 * np.pi, 0.01)
    x = []
    y = []
    for angles in angles_circle:
        or_x = a * cos(angles)
        or_y = b * sin(angles)
        length_or = sqrt(or_x * or_x + or_y * or_y)
        or_theta = atan2(or_y, or_x)
        new_theta = or_theta + e_angle/180*np.pi
        new_x = e_x + length_or * cos(new_theta)
        new_y = e_y + length_or * sin(new_theta)
        x.append(new_x)
        y.append(new_y)
    return x, y

e1 = Ellipse2d.Ellipse2d()
e2 = Ellipse2d.Ellipse2d()
e1._set_by_6pars(a=2, c=2, d=-5, f=2)
e2._set_by_6pars(a=2, c=2, d=-10, f=2)
e1._set_by_6pars(a=3, b=-4, c=2, d=-4, e=2, f=1)
e2._set_by_6pars(a=5, b=-6, c=2, d=-6, e=2, f=1)

x1,y1 = get_ellipse(e1.point2d_center[0], e1.point2d_center[1] ,e1.radius_u, e1.radius_v, e1.theta_rad)
x2,y2 = get_ellipse(e2.point2d_center[0], e2.point2d_center[1] ,e2.radius_u, e2.radius_v, e2.theta_rad)


ret = Ellipse2d.classify_relative_position_between_2ellipes(e1, e2)
plt.plot(x1, y1)
plt.plot(x2, y2, "r")
#plt.show()



test_case = [
    [[3, 2, 0, 0, 0], [3, 2, 0, 0, 0]],                         # 0
    [[3, 2, 0, 0, 0], [3, 1, 1, -0.5, np.pi/4]],                # 1
    [[3, 2, 0, 0, 0], [2, 1, -2, -1, np.pi/4]],                 # 2
    [[2, 1, 0, 0, 0], [1.5, 0.75, -2.5, 1.5, np.pi/4]],         # 3
    [[3, 2, 0, 0, 0], [2, 1, -0.75, 0.25, np.pi/4]],            # 4
    [[3, 2, 0, 0, 0], [2, 1, -0.75, 0.25, np.pi/4]],            # 5
    [[2, 1, 0, 0, 0], [2, 1, 0, 2, 0]],                         # 6
    [[3, 2, 0, 0, 0], [2, 1, -1.0245209260022, 0.25, np.pi/4]], # 7
    [[3, 2, 0, 0, 0], [1, 2, 0, 0, 0]],                         # 8
    [[9, 0, 100, 0, 0, -81], [143, 0, 773, -267, -155, -509]],  # 9
    
    ]

for i_test, test_pars in enumerate(test_case):
    print("\n-{}: {}/{}----------------------------------------------".format("case", i_test, len(test_case)))
    e1 = Ellipse2d.Ellipse2d()
    e2 = Ellipse2d.Ellipse2d()
    if   len(test_pars[0]) == 5:
        pars1 = np.array(test_case[i_test][0], dtype=float)
        pars2 = np.array(test_case[i_test][1], dtype=float)

        e1._set_by_5props(
                radius_u=pars1[0], 
                radius_v=pars1[1], 
                u_center=pars1[2], 
                v_center=pars1[3], 
                theta=pars1[4], 
            )
        print("pars 1: {:4f},\t{:4f},\t{},\t{:4f}".format(pars1[0], pars1[1], (pars1[2], pars1[3]), pars1[4]))
        print("props1: {:4f},\t{:4f},\t{},\t{:4f}".format(e1.radius_u, e1.radius_v, e1.point2d_center, e1.theta_rad))
        print()

        e2._set_by_5props(
                radius_u=pars2[0], 
                radius_v=pars2[1], 
                u_center=pars2[2], 
                v_center=pars2[3], 
                theta=pars2[4], 
            )
        print("pars 2: {:4f},\t{:4f},\t{},\t{:4f}".format(pars2[0], pars2[1], (pars2[2], pars2[3]), pars2[4]))
        print("props2: {:4f},\t{:4f},\t{},\t{:4f}".format(e2.radius_u, e2.radius_v, e2.point2d_center, e2.theta_rad))
        ret = Ellipse2d.classify_relative_position_between_2ellipes(e1, e2)
        if ret:
            print(i_test, ret.name, ret.value)
    elif len(test_case[0][0]) == 6:
        print()