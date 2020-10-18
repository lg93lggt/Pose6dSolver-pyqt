

import  cv2
import numpy as np
import  glob
import  os

if __name__ == "__main__":
    
    dir_images_1 = "/home/veily/桌面/OPENCV_AR-master/proj-simu/result/1"
    dir_images_2 = "/home/veily/桌面/OPENCV_AR-master/proj-simu/result/2"
    pths_imagse_1 = sorted(glob.glob(os.path.join(dir_images_1, "*.jpg")))
    pths_imagse_2 = sorted(glob.glob(os.path.join(dir_images_2, "*.jpg")))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    img_1 = cv2.imread(pths_imagse_1[0])
    [H, W] = img_1.shape[:2]

    video = cv2.VideoWriter('/home/veily/桌面/OPENCV_AR-master/proj-simu/result/match.avi', fourcc, 100.0, (W * 2, H),True)

    for idx in range(len(pths_imagse_1)):
        print("{:0>6d}".format(idx))
        img_for_video = np.zeros((H, W * 2, 3)).astype(np.uint8)
        img_1 = cv2.imread(pths_imagse_1[idx])
        img_2 = cv2.imread(pths_imagse_2[idx])
        img_for_video[:, :W, :] = img_1
        img_for_video[:, W:, :] = img_2
        video.write(img_for_video)
    video.release()
    #     if ret==True:

    #         cv2.imshow('frame',frame)
    #         out.write(frame)

    #         if cv2.waitKey(10) & 0xFF == ord('q'):
    #             break
    #     else:
    #         break

    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()