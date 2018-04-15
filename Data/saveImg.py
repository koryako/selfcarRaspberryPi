from time import sleep
from picamera import PiCamera

def open_preview():
    with PiCamera() as camera:
        camera.resolution = (320, 240)

        camera.start_preview()
        try:
          num=0
          while True:
             sleep(1)
             camera.capture('/media/pi/mov/img/img'+str(num)+'.jpg',use_video_port=True,resize=(80,60))
             num+=1
             if num==5:
                 break
        finally:
            camera.stop_preview()
if __name__ == '__main__':
    open_preview()

