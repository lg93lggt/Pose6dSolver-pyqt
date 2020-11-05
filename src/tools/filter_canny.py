
import cv2
import numpy as np
import glob
import os
import sys

def run(input_dir):
    out_dir = input_dir + "_canny"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    pths = glob.glob(input_dir + "/*/*.*")
    pths.sort()
    i = 0
    for pth in pths:
        i+=1
        name = os.path.split(pth)[1]
        img = cv2.imread(pth)
        edges = cv2.Canny(img, 100, 100)
        img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        print(out_dir + "/" + "{}.png".format(i))
        cv2.imwrite(out_dir + "/" + "{}.png".format(i), img)
        cv2.imshow("", img)
        cv2.namedWindow("", cv2.WINDOW_FREERATIO)
        key = cv2.waitKey()
        if key == ord("q"):
            break

if __name__ == "__main__":
    run(sys.argv[1])