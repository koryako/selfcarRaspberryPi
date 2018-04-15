import serial,os

port = os.popen('ls /dev/ttyACM*').read()[:-1]
baud = 115200
ser = serial.Serial(port, baud)

def getString():
   line = ser.readline()#.decode('utf-8')
   return line

def mian():
   while True:
      line=getString()
      print(line)



if __name__ == '__main__':
   main();















