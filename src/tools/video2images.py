import  cv2
import  os

def imwrite_with_zn(pth_image_with_zn, img):
    cv2.imencode('.png', img)[1].tofile(pth_image_with_zn)
    return 

def viedo2images(pth_video, dir_new=None, downsample=1):
    [dir_viedeo, name_video] = os.path.split(pth_video)
    [prefix_video, suffix_video] = os.path.splitext(name_video)

    if dir_new is None:
        dir_new = os.path.join(dir_viedeo, prefix_video)
        if not os.path.exists(dir_new):
            os.mkdir(dir_new)

    video = cv2.VideoCapture(pth_video)
    cnt = 0
    strat_reading =True
    while strat_reading:
        [strat_reading, img] = video.read()
        if strat_reading:
            if cnt % downsample == 0:
                imwrite_with_zn(os.path.join(dir_new, "{:0>6d}.png".format(cnt)), img)
                print(os.path.join(dir_new, "{:0>6d}.png".format(cnt)))
            cnt += 1
    video.release()
    return


if __name__ == "__main__":
    
    viedo2images("/home/nue/桌面/测试3/测试1.avi", "/home/nue/桌面/测试3/1", 1)
    viedo2images("/home/nue/桌面/测试3/测试2.avi", "/home/nue/桌面/测试3/2", 1)