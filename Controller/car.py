import RPi.GPIO as GPIO
import time
class Wheel:
  pins={'a':[18,16],'b':[35,37]}
  def __init__(self,name):
     self.name=name
     self.pin=Wheel.pins[self.name]
     GPIO.setmode(GPIO.BOARD)
     GPIO.setup(self.pin[0],GPIO.OUT)
     GPIO.setup(self.pin[1],GPIO.OUT)
     self.stop()
  def forward(self):
     GPIO.output(self.pin[0],True)
     GPIO.output(self.pin[1],False)
  def stop(self):
     GPIO.output(self.pin[0],False)
     GPIO.output(self.pin[1],False)
  def back(self):
     GPIO.output(self.pin[0],False)
     GPIO.output(self.pin[1],True)
class Car:
   wheels=[Wheel('a'),Wheel('b')]
   @staticmethod
   def init():
      GPIO.setmode(GPIO.BOARD)
   @staticmethod
   def forward():
      for wheel in Car.wheels:
          wheel.forward()
   @staticmethod
   def back():
      for wheel in Car.wheels:
          wheel.back()
   @staticmethod
   def left():
       Car.wheels[0].back()
       Car.wheels[1].forward()
   @staticmethod
   def left_p():
       Car.wheels[0].stop()
       Car.wheels[1].forward()
   @staticmethod
   def right():
       Car.wheels[0].forward()
       Car.wheels[1].back()
   @staticmethod
   def right_p():
       Car.wheels[0].forward()
       Car.wheels[1].stop()
   @staticmethod
   def stop():
       Car.wheels[0].stop()
       Car.wheels[1].stop()
       
commands={'forward':Car.forward,'back':Car.back,'stop':Car.stop,
  'left':Car.left,'right':Car.right,'right_p':Car.right_p,'left_p':Car.left_p}

def execute(command):
    print command  
    commands[command]()


def cleanup():
    GPIO.cleanup()

execute('forward')
