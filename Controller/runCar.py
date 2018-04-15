import serial,os
import readchar
import keypress
#from  keypress import *
#kp=KeyPress()
port=os.popen('ls /dev/ttyACM*').read()[:-1]
ser = serial.Serial(port, 9600)
kp=keypress.KeyPress()


forward=lambda:car.execute('forward')
back=lambda:car.execute('back')
left=lambda:car.execute('left_p')
right=lambda:car.execute('right_p')
stop=lambda:car.execute('stop')
fnServoUp   = lambda:ser.write('U')
fnServoDown = lambda:ser.write('D')


keymap={\
   "w":forward,\
   "s":back,\
   "a":left,\
   "d":right,\
   " ":stop,\
   'j':fnServoUp,\
   'k':fnServoDown,\
   's':fnHalt
}

kp.registerQuitKey('q')
kp.registerHandlers(keyMap)
kp.start()








