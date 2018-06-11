#! /usr/bin/python
# -*- coding:utf-8 -*-
import cv2
import numpy as np
from os import listdir
from os.path import isdir, isfile, join
import sys
from skew_detection import *
######################################
#               CLASSES                 #
######################################

class Image(object):
    def __init__(self, image, f = "", key = None, descriptor = None):

        '''
        #灰度化
        im_at_mean = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #二值化
        im_at_mean = cv2.adaptiveThreshold(im_at_mean, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 7)
        # 膨胀和腐蚀操作的核函数
        element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        # 腐蚀一次，去掉细节
        erosion = cv2.erode(im_at_mean, element1, iterations=1)
        # 膨胀，让轮廓明显一些
        dilation = cv2.dilate(erosion, element2, iterations=1)
        # 查找车牌区域
        region = findPlateNumberRegion(dilation)

        # 用绿线画出这些找到的轮廓
        #print region
        #for box in region:
        #    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
        showImage(im_at_mean)
        showImage(openingimg)
        showImage(dilation)
        #showImage(image)
    '''
        grayimg=final_license(image)
        self.img = grayimg
        #self.img = image
        self.fileName = f
        self.k = key
        self.d = descriptor
        self.cars = []

    def addCar(self, car):
        self.cars.append(car)



class Rectangle(object):
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        
######################################
#               FUNCTIONS             #
######################################
def printOK():
    print "\033[92mOK!\033[0m"

def printErrorMsg(text):
    """ Prints text to STDERR """
    print >> sys.stderr, text

def getInput(text):
    return (raw_input(text)).strip()

#def writeToFile(file, text):
#    f = open(file, 'w')
#    f.write(text)

def showImage(img):
    cv2.imshow('Matched Features', img)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')

def loadImgs(path):
    """ Given a Path, converts all images to grey scale and returns a list of Image objects """
    return [Image(cv2.imread(join(path, f),1), f) for f in listdir(path) if (isfile(join(path, f)) and f.endswith('.jpg'))]

def convertTupleListToRectangleList(l_xywh):
    """ Receives a list of tuples (x,y,w,h) defining rectangles
        Returns a list of Rectangle objects
    """
    l = []
    for (x,y,w,h) in l_xywh:
        l.append(Rectangle(x,y,w,h))
    return l


def final_license(image):
    # 压缩图像
    image = cv2.resize(image, (400, 400 * image.shape[0] / image.shape[1]))
    # RGB转灰色
    grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 图像二值化
    binaryimg = cv2.adaptiveThreshold(grayimg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 7)
    #showImage(binaryimg)
    # 使用Canny函数做边缘检测
    cannyimg = cv2.Canny(image, image.shape[0], image.shape[1])
    #showImage(cannyimg)
    # 消除小区域，保留大块区域，从而定位车牌
    # 进行闭运算
    kernel = np.ones((5, 19), np.uint8)
    closingimg = cv2.morphologyEx(cannyimg, cv2.MORPH_CLOSE, kernel)
    #showImage(closingimg)
    # 进行开运算
    openingimg = cv2.morphologyEx(closingimg, cv2.MORPH_OPEN, kernel)
    #showImage(openingimg)
    # 再次进行开运算
    kernel = np.ones((11, 5), np.uint8)
    openingimg = cv2.morphologyEx(openingimg, cv2.MORPH_OPEN, kernel)
    #showImage(openingimg)
    # 定位车牌
    rect = locate_license(openingimg, image)
    # 在image上标出车牌范围
    cv2.rectangle(image, (rect[0] - 10, rect[1] - 10), (rect[2] + 10, rect[3] + 10), (0, 255, 0), 2)
    # 截取车牌图像
    plateimg = binaryimg[rect[1] - 10:rect[3] + 10, rect[0] - 10:rect[2] + 10]
    # 倾斜矫正
    skew_h, skew_v = skew_detection(plateimg)
    corr_img = v_rot(plateimg, (90 - skew_v + skew_h), plateimg.shape, 60)
    corr_img = h_rot(corr_img, skew_h)
    # 显示原始图像
    showImage(image)
    # 显示车牌图像
    showImage(plateimg)
    # 显示倾斜矫正后的车牌图像
    showImage(corr_img)

    return grayimg

def find_retangle(contour):
    y, x = [], []
    for p in contour:
        y.append(p[0][0])
        x.append(p[0][1])
    return [min(y), min(x), max(y), max(x)]


def locate_license(img, orgimg):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找出最大的三个区域
    blocks = []
    for c in contours:
        # 找出轮廓的左上点和右下点，由此计算它的面积和长宽比
        r = find_retangle(c)
        a = (r[2] - r[0]) * (r[3] - r[1])
        s = (r[2] - r[0]) / (r[3] - r[1])
        # 车牌正常情况下长高比在2.7-5之间
        if (s > 5 or s < 2):
            continue
        blocks.append([r, a, s])

        # 选出面积最大的3个区域
    blocks = sorted(blocks, key=lambda b: b[2])[-3:]

    # 使用颜色识别判断找出最像车牌的区域
    maxweight, maxinedx = 0, -1
    for i in xrange(len(blocks)):
        b = orgimg[blocks[i][0][1]:blocks[i][0][3], blocks[i][0][0]:blocks[i][0][2]]
        # RGB转HSV
        hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
        # 蓝色车牌范围
        lower = np.array([100, 50, 50])
        upper = np.array([140, 255, 255])
        # 根据阈值构建掩模
        mask = cv2.inRange(hsv, lower, upper)
        # 统计权值
        w1 = 0
        for m in mask:
            w1 += m / 255
        w2 = 0
        for w in w1:
            w2 += w
        # 选出最大权值的区域
        if w2 > maxweight:
            maxindex = i
            maxweight = w2
    return blocks[maxindex][0]


