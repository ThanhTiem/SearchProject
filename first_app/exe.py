import cv2
import numpy as np
def readImg(path):
    print(path)
    img = cv2.imread("media/ant_0014.jpg")
    print(type(img))
    image = path.split('/')
    return "static/dataset2/" + image[2]
