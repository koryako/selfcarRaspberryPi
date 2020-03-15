import matplotlib.pyplot as plt
import eventlet
import socketio
import cv2
import base64
import numpy as np
from PIL import Image
from io import BytesIO
sio = socketio.Server()
app = socketio.WSGIApp(sio, static_files={
    '/': {'content_type': 'text/html', 'filename': 'index.html'}
})

@sio.event
def connect(sid, environ):
    print('connect ', sid)

@sio.event
def my_message(sid, data):
    if data:
        #print(data)
        # The current steering angle of the car
        #steering_angle = data["steering_angle"]
        # The current throttle of the car
        #throttle = data["throttle"]
        # The current speed of the car
        #speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        #img_array = np.fromstring(base64.b64decode(imgString),np.uint8) # 转换np序列
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        #image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        img = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
        #img_array= np.asarray(image)
        cv2.imwrite('test.jpg',img,[int(cv2.IMWRITE_JPEG_QUALITY),70])
        #cv2.imshow("Frame", img)
        #print(img_array)
    
@sio.event
def disconnect(sid):
    print('disconnect ', sid)

if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)