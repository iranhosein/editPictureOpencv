import cv2
import matplotlib.pyplot as plt
import sys
import os
import argparse
import time
import numpy as np
# import pandas as pd
# from PyQt6.QtWidgets import QApplication, QWidget

mouseFlag = False

def readImage(image_address):
    img = cv2.imread(image_address)
    cv2.imshow('Your Picture', img)
    cv2.waitKey(0)

def cropImage(img, x_start, x_finish, y_start, y_finish):
    return img[x_start:x_finish, y_start:y_finish]

def rotationImage(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

def flipImage(img, horizontal_or_vertical):
    return cv2.flip(img, int(horizontal_or_vertical == 'horizontal'))

def blurImage(img, kernel_x, kernel_y):
    return cv2.GaussianBlur(img, (kernel_x, kernel_y), 0)

def borderImage(img, border, borderSize):
    return cv2.copyMakeBorder(src=img, top=borderSize, bottom=borderSize, left=borderSize, 
                              right=borderSize, borderType=cv2.BORDER_CONSTANT)

def colorImage(img, colors):
    np.flip(colors)
    for i in range(0,3):
        img[:,:,i] = np.where(img[:,:,i]+colors[i]>=0, 
                              np.where(img[:,:,i]+colors[i]<=255, img[:,:,i]+colors[i], 255), 0)
    return img

def brightnessImage(img, brightness):
    return cv2.convertScaleAbs(img, 1, brightness)

def contrastImage(img, contrast):
    return cv2.addWeighted(img, 1, img, 0, contrast)

def combineImage(img1, percentImg1, img2):
    width = min(int(img1.shape[1]), int(img2.shape[1]))
    height = min(int(img1.shape[0]), int(img2.shape[0]))
    dim = (width, height)
    img1 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)
    return cv2.addWeighted(img1, percentImg1, img2, 1-percentImg1, 0)

def downscaleImage(img, number):
    width = int(img.shape[1]/number)
    height = int(img.shape[0]/number)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

def upscaleImage(img, number):
    width = int(img.shape[1]*number)
    height = int(img.shape[0]*number)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

def concatImage(img, number):
    num = int(np.sqrt(number))
    width = int(img.shape[1]/num)
    height = int(img.shape[0]/num)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    arr = []
    for i in range(0, num):
        arr.append(img)
    img = cv2.vconcat(arr)
    arr = []
    for i in range(0, num):
        arr.append(img)
    img = cv2.hconcat(arr)
    return img

def decreaseResolution(img, number):
    first_width = img.shape[1]
    first_height = img.shape[0]
    width = int(first_width/number)
    height = int(first_height/number)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    dim = (first_width, first_height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

def increaseResolution(img, scale):
    filemodel_filepath = 'D:/image processing/pretrained_models/EDSR_x4.pb'
    model_pretrained = cv2.dnn_superres.DnnSuperResImpl_create()
    model_pretrained.readModel(filemodel_filepath)
    model_pretrained.setModel('edsr', scale)
    return model_pretrained.upsample(img)

def setCoordinates(coordinates, x, y):
    #[x_start, x_finish, y_start, y_finish]
    if x < coordinates[0]:
        coordinates[0] = x
    elif x > coordinates[1]:
        coordinates[1] = x
    if y < coordinates[2]:
        coordinates[2] = y
    elif y > coordinates[3]:
        coordinates[3] = y
    return coordinates

def click_event(event, x, y, flags, params):
    global mouseFlag, point
    #[x_start, x_finish, y_start, y_finish]
    coordinates = [0.0, 0.0, 0.0, 0.0]
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseFlag = True
        cv2.circle(img, (x,y), 3, (0, 255, 255), -1)
        setCoordinates(coordinates, x, y)
        point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        mouseFlag = False
        setCoordinates(coordinates, x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if mouseFlag==True:
            cv2.line(img, point, (x,y), (0, 255, 255), thickness=3)
            point = (x, y)

def draw(img):
    cv2.namedWindow('draw')
    cv2.setMouseCallback('draw', click_event)
    while True:
        cv2.imshow('draw', img)
        k = cv2.waitKey(1) & 0xFF
        if k==27:
            break
    cv2.destroyAllWindows()
    

def delObjImage(img):
    
    coordinates = [0.0, 0.0, 0.0, 0.0]


image_address = 'D:/very test/aaaa/kitten.jpg'
img = cv2.imread(image_address)
# cv2.imshow('Your Picture', img)
# cropImage(img, 10,280,20,360)
# cv2.imshow('rotated image', rotationImage(rotationImage(img)))
# cv2.imshow('flip image', flipImage(img, 'horizontal'))
# cv2.imshow('flip image', flipImage(img, 'vertical'))
# cv2.imshow('blur image', blurImage(img, 35, 35))
# cv2.imshow('border image', borderImage(img, 35, 15))
# cv2.imshow('color image1', colorImage(img, [-10, 0, 0]))
# cv2.imshow('color image100', brightnessImage(img, 0.1))
# cv2.imshow('color image100', contrastImage(img, -40))
# image1_address = 'D:/very test/aaaa/1.jpg'
# img1 = cv2.imread(image1_address)
# image2_address = 'D:/very test/aaaa/tools.jpg'
# img2 = cv2.imread(image2_address)
# cv2.imshow('color image100', combineImage(img, 0.5, img2))
# cv2.imshow('color image100', increaseResolution(img, 4))
# cv2.imshow('color image100', increaseResolution(img, 4))
# cv2.imshow('color image100', decreaseResolution(img, 10))
# draw(img)
# cv2.imshow('color image100', draw(img))
cv2.imshow('color image100', increaseResolution(img, 4))



# start = time.time()
# threshold = 0.4
# image = cv2.imread('D:/very test/aaaa/lag1.jpg')
# cv2.imshow('jlkjfdf', image)
# imageG = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# #template = cv2.imread('template.jpg', 0)
# template = cv2.imread('D:/very test/aaaa/template1.png', 0)
# w, h = template.shape[::-1]
# res = cv2.matchTemplate(imageG,template,cv2.TM_CCOEFF_NORMED)
# loc = np.where( res >= threshold)


# mask = np.zeros_like(imageG)
# for pt in zip(*loc[::-1]):
#     #a = cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,0,255), 1)
#     cv2.rectangle(mask, (pt[0]+3, pt[1]+3), (pt[0]+w-3, pt[1]+h-3), 255, -1)
#     # Reduce the size of the rectangle by 3 pixels from each side


# image = cv2.inpaint(image, mask, 2, cv2.INPAINT_NS)

# cv2.imshow('lag.jpg', image)
# cv2.imshow('mask', mask)
# end = time.time()
# print(end - start)
# cv2.waitKey(0)
# cv2.destroyAllWindows()














# cv2.imshow('color image100', concatImage(img, 32))
# cv2.imshow('color image0', img[:,:,0])
# cv2.imshow('color image2', img[:,:,2])
# img[:,:,0] = 0
# img[:,:,2] += 200
# img[:,:,2] += 200
# print(img)
# cv2.imshow('color4', img)
cv2.waitKey(0)

# image_address = input("Please enter your image address: ")
# image_address = 'D:/very test/aaaa/kitten.jpg'
# img = cv2.imread(image_address)
# cv2.imshow('Your Picture', img)
# cv2.waitKey(0)

# print(image_address)

# app = QApplication(sys.argv)

# window = QWidget()
# window.show()

# app.exec()