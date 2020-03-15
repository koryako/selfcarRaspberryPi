import socketio
import cv2
import base64
sio = socketio.Client()
vs = cv2.VideoCapture(0)

def send_image():
    while vs.isOpened():
        ok,frame = vs.read()
        #cv2.imshow("Frame", frame)
        img_str = cv2.imencode('.jpg', frame)[1].tostring()  # 将图片编码成流数据，放到内存缓存中，然后转化成string格式
        b64_code = base64.b64encode(img_str) # 编码成base64 .encode('utf-8')
        #print(b64_code)
        sio.emit('my_message', {'image': b64_code})
        key = cv2.waitKey(1) & 0xFF
        if key == ord('k') or key == 'k':
            cv2.imwrite(p, image)
        elif key == ord('q') or key == 'q':
            break
    print("[INFO] {} face images stored".format(total))
    print("[INFO] cleaning up...")
    cv2.destroyAllWindows()
    vs.release()

@sio.event
def connect():
    print('connection established')
    sio.emit('my_message')
    





@sio.event
def my_message(data):
    print('message received with ', data)
    

   
@sio.event
def disconnect():
    print('disconnected from server')

if __name__ == '__main__':
    sio.connect('http://localhost:5000')
    send_image()
    #sio.wait()
