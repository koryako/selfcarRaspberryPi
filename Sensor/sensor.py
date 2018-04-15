import RPi.GPIO as GPIO
import time
class s:
  def __init__(self):
     self.init() 
     self.dist=0    
  def checkdist(self):
     GPIO.output(22,GPIO.HIGH)
     time.sleep(0.000015)
     GPIO.output(22,GPIO.LOW)
     while not GPIO.input(40):
         pass
     t1=time.time()
     while GPIO.input(40):
         pass
     t2=time.time()
     return (t2-t1)*340/2
   
  def init(self):
     GPIO.setmode(GPIO.BOARD)  
     GPIO.setup(22,GPIO.OUT,initial=GPIO.LOW)
     GPIO.setup(40,GPIO.IN)

  def cleanup(self):
     GPIO.cleanup()


  def start(self):
    self.init()
    time.sleep(2)
    while True:
       
       self.dist=self.checkdist()*100
              
       time.sleep(1)
    self.clearnup()

if __name__=="__main__":
     sens=s()
     # sens.start()
     
     print sens.checkdist()*100
     sens.cleanup()
