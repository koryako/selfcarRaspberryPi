import RPi.GPIO as GPIO
import time
import sys
import Tkinter as tk

def init():    
   GPIO.setmode(GPIO.BOARD)
   GPIO.setup(18,GPIO.OUT)
   GPIO.setup(16,GPIO.OUT)
   GPIO.setup(35,GPIO.OUT)
   GPIO.setup(37,GPIO.OUT)   
   
def zuolun(e1,e2):
   GPIO.output(18,e1)
   GPIO.output(16,e2)

def youlun(e1,e2):
   GPIO.output(35,e1)
   GPIO.output(37,e2)

def forward(s): 
   init() 
   zuolun(True,False)
   youlun(True,False)
   time.sleep(s)
   GPIO.cleanup()
   
def back(s):
   init()
   zuolun(False,True)
   youlun(False,True)
   time.sleep(s)
   GPIO.cleanup()

def stop(s):
   zuolun(False,False)
   youlun(False,False)
   time.sleep(s)
   GPIO.cleanup()

def left(s):
   zuolun(False,True)
   youlun(True,False)
   time.sleep(s)
   GPIO.cleanup()

def right(s): 
   zuolun(True,False)
   youlun(False,True)
   time.sleep(s)
   GPIO.cleanup()

def sensInit():
   GPIO.setmode(GPIO.BOARD)
   GPIO.setup(22,GPIO.OUT,initial=GPIO.LOW)
   GPIO.setup(40,GPIO.IN)

def distance(measure='cm'):
   GPIO.setmode(GPIO.BOARD)
   GPIO.setup(22,GPIO.OUT)
   GPIO.setup(40,GPIO.IN)
   GPIO.output(22,False)
   while GPIO.input(40)==0:
       nosig=time.time()
   while GPIO.input(40)==1:
       sig=time.time()
   t1=sig-nosig
   if measure=='cm':
      distance=t1/0.000058
   elif measure=='in':
      distance=t1/0.000148
   else:
      print ('error')
      distance=None

   GPIO.clearup()
   return distance

def checkdist():
   GPIO.output(22,GPIO.HIGH)
   time.sleep(0.000015)
   GPIO.output(22,GPIO.LOW)
   while not GPIO.input(40):
        pass
   t1=time.time()
   while not GPIO.input(40):
        pass
   t2=time.time()
   return (t2-t1)*340/2


"""
def key_input(event):
    
    print event.char
    key_press=event.char
    sleep_time=0.030
    
    if key_press.lower()=='w':
        car.execute('forward')
        
    elif key_press.lower()=='s':
        car.execute('back')
    elif key_press.lower()==' ':
        car.execute('stop')
    elif key_press.lower()=='d':
        car.execute('right_p')
    elif key_press.lower()=='a':
        car.execute('left_p')
    else:
        pass

"""






def key_input(event):
   init()
   
   print 'Key:',event.char
   key_press=event.char
   sleep_time=0.030
   if key_press.lower()== 'w':
        forward(sleep_time)
   elif key_press.lower()== 's':
        back(sleep_time)
   elif key_press.lower()=='d':
        right(sleep_time)
   elif key_press.lower()=='a':
        left(sleep_time)
   """   
   else:
        pass
   sensInit()
   curDis=distance('cm')*100
   print('curdis is',curDis)
   """

if __name__=="__main__":
   command=tk.Tk()
   command.bind('<KeyPress>',key_input)
   command.mainloop()
   GPIO.cleanup()
