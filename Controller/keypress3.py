import sys
import time
#import serial,os
"""
port = os.popen('ls /dev/ttyACM*').read()[:-1]
baud = 9600
ser = serial.Serial(port, baud)

#while True:
    #line = ser.readline()#.decode('utf-8')
    #print line
"""


    
class KeyPress:
    def __init__(self):
        def getch():   # define non-Windows version
            import tty, termios
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch
        self._getch = getch
        self._char = None
        self._quitKey = None
        self._handlersMap = {}

    def registerHandlers(self,hMap):
        self._handlersMap = hMap;

    def registerQuitKey(self,char):
        self._quitKey = char

    def start(self):
        while True:
            #line = ser.readline()#.decode('utf-8')
            #self._char=line.strip("\n")
            self._char = self._getch()
            if self._char == self._quitKey:
                print "QUIT!"
                return
            if self._char in self._handlersMap:
                self._handlersMap[self._char]()
            
