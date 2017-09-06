import numpy as np
import cv2
import math
from scipy import ndimage

class ConvertImage(object):
    '''
    Convert image to fixed size for mnist
    '''
    def __init__(self, width, height):
        '''
        width, height 是目标图像的宽和高
        '''
        self.width = width
        self.heigth = height

    def convert2mnist(self, src):
        '''
        将width*height的图像src转换为适合mnist输入的向量（一般是28*28的图像转为长度为784的向量）
        '''
        (thresh, gray) = cv2.threshold(src, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        gray = cv2.resize(255-gray,(self.width,self.heigth))  

        while np.sum(gray[0]) == 0:
            gray = gray[1:]

        while np.sum(gray[:,0]) == 0:
            gray = np.delete(gray,0,1)

        while np.sum(gray[-1]) == 0:
            gray = gray[:-1]

        while np.sum(gray[:,-1]) == 0:
            gray = np.delete(gray,-1,1)

        rows,cols = gray.shape

        if rows>cols:
            factor = 20.0/rows
            rows = 20
            cols = int(round(cols*factor))
            gray = cv2.resize(gray,(cols,rows))
        else:
            factor = 20.0/cols
            cols = 20
            rows = int(round(rows*factor))
            gray = cv2.resize(gray,(cols,rows))

        colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
        rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
        gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

        shiftx,shifty = self.getBestShift(gray)
        shifted = self.shift(gray,shiftx,shifty)
        gray = shifted

        arr = []

        for i in range(28):
            for j in range(28):
                pixel = float(gray[i, j])/255.0
                arr.append(pixel)

        arr1 = np.array(arr).reshape(784, 1)
        tmp = np.asarray(arr1)
        return tmp        

    def getBestShift(self, img):
        cy,cx = ndimage.measurements.center_of_mass(img)
        rows,cols = img.shape
        shiftx = np.round(cols/2.0-cx).astype(int)
        shifty = np.round(rows/2.0-cy).astype(int)
        return shiftx,shifty

    def shift(self, img, sx, sy):
        rows,cols = img.shape
        M = np.float32([[1,0,sx],[0,1,sy]])
        shifted = cv2.warpAffine(img,M,(cols,rows))
        return shifted
