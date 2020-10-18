import cv2
import numpy as np

def img_contrast_brihtness(img, contrast, brightness):
    '''改变对比度和亮度'''
    dst = np.uint8(np.clip((contrast * img + brightness), 0, 255))
    return dst

def img_saturability(img):
    '''
    饱和度
    '''
    # 图像归一化，且转换为浮点型
    fImg = img.astype(np.float32)
    fImg = fImg / 255.0

    # 颜色空间转换 BGR转为HLS
    hlsImg = cv2.cvtColor(fImg, cv2.COLOR_BGR2HLS)


def img_color():
    pass