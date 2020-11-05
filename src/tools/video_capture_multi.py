import cv2
import numpy as np
import  sys
import os


if __name__ == '__main__':
    args = sys.argv
    dir_output = args[0]

    n_cams = 3
    cam1 = cv2.VideoCapture(0)
    cam2 = cv2.VideoCapture(4)
    cam3 = cv2.VideoCapture(2)
    
    dir_output = "/home/veily/桌面/1005"
    dirs_output = []
    for i_cam in range(n_cams):
        dir_new = os.path.join(dir_output, str(i_cam+1))
        if not os.path.exists(dir_new):
            os.mkdir(dir_new)
        dirs_output.append(dir_new)

    cnt = 0
    while True:
        ret1, img1 = cam1.read()
        ret2, img2 = cam2.read()
        ret2, img3 = cam3.read()
        if ret1 and ret2:
            cv2.imshow("1", img1)
            cv2.imshow("2", img2)
            cv2.imshow("3", img3)
            key = cv2.waitKey(1)

            if key == ord("c"):
                for i_cam in range(n_cams):
                    img = [img1, img2, img3][i_cam]
                    pth_image = dirs_output[i_cam] + "/{:0>3d}_{:d}.png".format(cnt, i_cam)
                    cv2.imwrite(pth_image, img)
                    print(pth_image)
                cnt += 1