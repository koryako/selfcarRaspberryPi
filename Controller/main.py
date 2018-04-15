#import keypress
from libs import forward,back,right,left
import serial,os
port = os.popen('ls /dev/ttyACM*').read()[:-1]
baud = 9600
ser = serial.Serial(port, baud)


#kp=keypress.KeyPress()
sleep_time=0.030
def w():
    forward(sleep_time)
    print "w"
     
def s():
    back(sleep_time)
    print "s"

def r():
    right(sleep_time)
    print "d"

def l():
    left(sleep_time)
    print "a"    
fnForward=lambda:w()
fnBack=lambda:s()
fnLeft=lambda:l()
fnRight=lambda:r()

keyMap = {\
    'w':fnForward,\
    'd':fnRight,\
    'a':fnLeft,\
    's':fnBack,\
}

while True:
    line = ser.readline()#.decode('utf-8')
    line = line.strip("\n")
    line = line.strip()
    print line
    if line=="w":
       w()
    else:
       s()
    
"""
kp.registerQuitKey('q')
kp.registerHandlers(keyMap)
kp.start()
"""
