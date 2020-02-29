#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
from base_camera import BaseCamera
import imutils
import time
import numpy
class Camera(BaseCamera):
    video_source = 0
    @staticmethod
    def set_video_source(source):
        Camera.video_source = source
    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        camera.set(3,160)
        camera.set(4,160)
        avg = None
        def nothing(x):
            pass
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')
        while True:
            # read current frame
            _, img = camera.read()
            cv2.resize(img,(img.shape[1]//2,img.shape[0]//2),interpolation=cv2.INTER_CUBIC)
            #img = imutils.resize(img, width=500)
            #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #高斯滤波
            #gray = cv2.blur(gray,(kerne,kerne))
            #gray = cv2.GaussianBlur(gray, (21,21),0)
            #gray=cv2.medianBlur(gray,kerne)
            # 如果平均帧是None，初始化它
            #if avg is None:
                #avg = gray.copy().astype("float")
                #continue
                #背景更新，调节第三个参数改变权重,改变被景更新速度
            #cv2.accumulateWeighted(gray, avg, 0.5)
            #获取两福图的差值---cv2.convertScaleAbs()--用来转换avg为uint8格式
            #frameDelta = cv2.absdiff(gray,cv2.convertScaleAbs(avg))
            # 对变化的图像进行阈值化，膨胀阈值图像来填补
            #thresh = cv2.threshold(frameDelta,5,255,cv2.THRESH_BINARY)[1]
            #膨胀图像
            #thresh = cv2.dilate(thresh,None,iterations=2)
            #寻找轮廓，存储在cnts里面
            #_, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            # 遍历轮廓线
            #for c in cnts:
                #print(cv2.contourArea(c))#打印检测运动区域大小
                #调整这个值的大小可以调整检测动作的大小
                #if cv2.contourArea(c) > 1000:
                    #(x, y, w, h) = cv2.boundingRect(c)#计算轮廓线的外框，为当前帧画上外框
                    #cv2.rectangle(img,(x,y),(x+w,y+h), (0, 255, 0),1)# 更新文本
            #cv2.putText(img, "action-detector", (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            yield numpy.array(cv2.imencode('.jpg', img)[1]).tostring() 
