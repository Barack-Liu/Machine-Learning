import cv2
import numpy as np
from os import listdir
from os.path import isdir, isfile, join


class Image(object):
	def __init__(self, image, f = "", key = None, descriptor = None):
		self.img = image
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
		

def printOK():
    print "\033[92mOK!\033[0m" 
def printErrorMsg(text):
	""" Prints text to STDERR """
	print >> stderr, text
def getInput(text):
	return (raw_input(text)).strip()


def showImage(img):
	cv2.imshow('Matched Features', img)
	cv2.waitKey(0)
	cv2.destroyWindow('Matched Features')
def loadImgs(path):
	""" Given a Path, converts all images to grey scale and returns a list of Image objects """
	return [Image(cv2.imread(join(path, f),0), f) for f in listdir(path) if (isfile(join(path, f)) and f.endswith('.jpg'))]
def convertTupleListToRectangleList(l_xywh):
    """ Receives a list of tuples (x,y,w,h) defining rectangles
        Returns a list of Rectangle objects
    """
    l = []
    for (x,y,w,h) in l_xywh:
        l.append(Rectangle(x,y,w,h))
    return l
