# -*- coding: utf-8 -*-
###################################################
#               智能小车1.0 -- 电机模块
#
#               @author chenph
#               @date 2018/5/10
###################################################

import RPi.GPIO as GPIO
import time


class WheelModule:

    # 初始化
    def __init__(self, PIN_L,PIN_IN1_L, PIN_IN2_L,PIN_R ,PIN_IN1_R, PIN_IN2_R):
        print('Wheel Module In Progress')
        GPIO.setmode(GPIO.BOARD)
        self.PIN_IN1_L = PIN_IN1_L
        self.PIN_IN2_L = PIN_IN2_L
        self.PIN_IN1_R = PIN_IN1_R
        self.PIN_IN2_R = PIN_IN2_R
        GPIO.setup(self.PIN_IN1_L, GPIO.OUT)
        GPIO.setup(self.PIN_IN2_L, GPIO.OUT)
        GPIO.setup(self.PIN_IN1_R, GPIO.OUT)
        GPIO.setup(self.PIN_IN2_R, GPIO.OUT)
        self.motor_L = GPIO.PWM(PIN_L,100)
        self.motor_L.start(100)
        self.motor_R = GPIO.PWM(PIN_R,100)
        self.motor_R.start(100)
        
    # 前进的代码
    def forward(self):
        GPIO.output(self.PIN_IN1_L, GPIO.HIGH)
        GPIO.output(self.PIN_IN2_L, GPIO.LOW)
        GPIO.output(self.PIN_IN1_R, GPIO.HIGH)
        GPIO.output(self.PIN_IN2_R, GPIO.LOW)

    # 后退
    def backOff(self):
        GPIO.output(self.PIN_IN1_L, GPIO.LOW)
        GPIO.output(self.PIN_IN2_L, GPIO.HIGH)
        GPIO.output(self.PIN_IN1_R, GPIO.LOW)
        GPIO.output(self.PIN_IN2_R, GPIO.HIGH)
    
    
    def forward_S(self,speed):
        if speed>0 :
            if speed > 100 :
                speed = 100
            self.motor_L.ChangeDutyCycle(speed)
            self.motor_R.ChangeDutyCycle(speed)
            GPIO.output(self.PIN_IN1_L, GPIO.HIGH)
            GPIO.output(self.PIN_IN2_L, GPIO.LOW)
            GPIO.output(self.PIN_IN1_R, GPIO.HIGH)
            GPIO.output(self.PIN_IN2_R, GPIO.LOW)
        
            
            
            

    # 左转
    def leftTurn(self):
        GPIO.output(self.PIN_IN1_L, GPIO.HIGH)
        GPIO.output(self.PIN_IN2_L, GPIO.LOW)
        GPIO.output(self.PIN_IN1_R, GPIO.LOW)
        GPIO.output(self.PIN_IN2_R, GPIO.LOW)

    # 右转
    def rightTurn(self):
        GPIO.output(self.PIN_IN1_L, GPIO.LOW)
        GPIO.output(self.PIN_IN2_L, GPIO.LOW)
        GPIO.output(self.PIN_IN1_R, GPIO.HIGH)
        GPIO.output(self.PIN_IN2_R, GPIO.LOW)

    # 停止
    def stop(self):
        GPIO.output(self.PIN_IN1_L, GPIO.LOW)
        GPIO.output(self.PIN_IN2_L, GPIO.LOW)
        GPIO.output(self.PIN_IN1_R, GPIO.LOW)
        GPIO.output(self.PIN_IN2_R, GPIO.LOW)


if __name__ == "__main__":
    try:
        m = WheelModule(16, 18, 35, 37)
        m.forward()
        time.sleep(5)
        m.stop()
        time.sleep(1)

        m.backOff()
        time.sleep(5)
        m.stop()
        time.sleep(1)

        m.leftTurn()
        time.sleep(2)
        m.stop()
        time.sleep(1)

        m.rightTurn()
        time.sleep(2)
        m.stop()
    except KeyboardInterrupt:
        pass
    GPIO.cleanup()