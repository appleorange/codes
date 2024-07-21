from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, imsave
import numpy as np
import shutil
from scipy import ndimage
from time import perf_counter

import os
from os import listdir

##import cv2


def sobel(im):
    im = im.astype('uint32')
    dx = ndimage.sobel(im, 1)  # horizontal derivative
    dy = ndimage.sobel(im, 0)  # vertical derivative
    mag = np.sqrt(dx ** 2 + dy ** 2)  # magnitude
    mag *= 255.0 / np.max(mag)  # normalize (Q&D)
    return mag.astype(np.uint8)


def main():
    #root = "/Users/yinghong_imac/Sabella Research Project/cropped_/cropped_image" #Root Path of Images
    #new_root = "/Users/yinghong_imac/Sabella Research Project/cropped_/contour_ouputs" #New Root Path of Contour Images
    
    root = "\\Users\\yulia\\code\\UIUC\\data65k\\test\\jpeg" #Root Path of Images
    new_root = "\\Users\\yulia\\code\\UIUC\\data65k\\test\\image" #New Root Path of Contour Images
    for img in listdir(root):
        img_path = root + os.sep + img
        new_img_path = new_root + os.sep + img
        im = Image.open(img_path, 'r')
        im = np.asarray(im)
        contour_img = sobel(im)
        imsave(new_img_path, contour_img)



if __name__ == '__main__':
    main()
