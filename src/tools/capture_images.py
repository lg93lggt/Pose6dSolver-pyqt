import cv2
import numpy as np
import os

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def capture_images(save_path):
    pths1 = save_path + '/1'
    pths2 = save_path + '/2'

    make_dir(pths1)
    make_dir(pths2)

    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(4)

    width = 640 
    height  = 480 

    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    num = 1
    while(cap1.isOpened()):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        k = cv2.waitKey(100) 
        # frame1_ = cv2.resize(frame1, (width, height))
        # frame2_ = cv2.resize(frame2, (width, height))
        if ret1 and ret2 :
            frames = np.hstack((frame1, frame2))
            cv2.imshow('1', frame1)
            cv2.imshow('2 ', frame2)
            print("num", num)
            #cv2.imshow('show_img: ', frames)
            # cv2.imwrite(pths1 + '/00{}_0.jpg'.format(num), frame1)
            # cv2.imwrite(pths2 + '/00{}_1.jpg'.format(num), frame2)
            cv2.imwrite(pths1 + '/00{}_0.jpg'.format(num), frame1)
            cv2.imwrite(pths2 + '/00{}_1.jpg'.format(num), frame2)
            print('保存成功')
            num += 1
        
        elif k == ord('q'):
            break
        print(k)
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    save_path  = '/media/veily/Data/LiGan/imgs'
    capture_images(save_path)
