#!coding=utf8
import io
import socket
import struct
import time
import picamera
# create socket and bind host
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.0.100', 8000))
connection = client_socket.makefile('wb')
 
try:
    with picamera.PiCamera() as camera:
        camera.resolution = (320, 240)      # pi camera resolution
        camera.framerate = 15               # 15 frames/sec
        time.sleep(2)                       # give 2 secs for camera to initilize
        start = time.time()
        stream = io.BytesIO()
        
        # send jpeg format video stream
        for foo in camera.capture_continuous(stream, 'jpeg', use_video_port = True):
            connection.write(struct.pack('<L', stream.tell()))
            connection.flush()
            stream.seek(0)
            connection.write(stream.read())
            if time.time() - start > 600:
                break
            stream.seek(0)
            stream.truncate()
    connection.write(struct.pack('<L', 0))
finally:
    connection.close()
    client_socket.close()



#https://blog.csdn.net/rebelqsp/article/details/22109925


"""
#!coding=utf8
import socket
import cv2
import threading
import struct
import numpy
 
class Camera_Connect_Object:
    def __init__(self,D_addr_port=["",8881]):
        self.resolution=[640,480]
        self.addr_port=D_addr_port
        self.src=888+15                 #双方确定传输帧数，（888）为校验值
        self.interval=0                 #图片播放时间间隔
        self.img_fps=15                 #每秒传输多少帧数
 
    def Set_socket(self):
        self.client=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.client.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
 
    def Socket_Connect(self):
        self.Set_socket()
        self.client.connect(self.addr_port)
        print("IP is %s:%d" % (self.addr_port[0],self.addr_port[1]))
 
    def RT_Image(self):
        #按照格式打包发送帧数和分辨率
        self.name=self.addr_port[0]+" Camera"
        self.client.send(struct.pack("lhh", self.src, self.resolution[0], self.resolution[1]))
        while(1):
            info=struct.unpack("lhh",self.client.recv(12))
            buf_size=info[0]                    #获取读的图片总长度
            if buf_size:
                try:
                    self.buf=b""                #代表bytes类型
                    temp_buf=self.buf
                    while(buf_size):            #读取每一张图片的长度
                        temp_buf=self.client.recv(buf_size)
                        buf_size-=len(temp_buf)
                        self.buf+=temp_buf      #获取图片
                        data = numpy.fromstring(self.buf, dtype='uint8')    #按uint8转换为图像矩阵
                        self.image = cv2.imdecode(data, 1) 
                        print(self.image.shape)                 #图像解码
                        #cv2.imshow(self.name, self.image)                   #展示图片
                except:
                    pass;
                finally:
                    if(cv2.waitKey(10)==27):        #每10ms刷新一次图片，按‘ESC’（27）退出
                        self.client.close()
                        cv2.destroyAllWindows()
                        break
 
    def Get_Data(self,interval):
        showThread=threading.Thread(target=self.RT_Image)
        showThread.start()
 
if __name__ == '__main__':
    camera=Camera_Connect_Object()
    camera.addr_port[0]="192.168.0.100"
    camera.addr_port=tuple(camera.addr_port)
    camera.Socket_Connect()
    camera.Get_Data(camera.interval)


#查看某个端口是否被占用: lsof -i:端口号
#https://blog.csdn.net/rebelqsp/article/details/22109925

"""
#!coding=utf-8
"""
import socket
import os
import sys
import struct

def socket_client():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('127.0.0.1', 9001))
    except socket.error as msg:
        print (msg)
        sys.exit(1)
    print (s.recv(1024))

    # 需要传输的文件路径
    filepath = 'jpg/augmented-image-example.png'
    # 判断是否为文件
    if os.path.isfile(filepath):
        # 定义定义文件信息。128s表示文件名为128bytes长，l表示一个int或log文件类型，在此为文件大小
        fileinfo_size = struct.calcsize('128sl')
        # 定义文件头信息，包含文件名和文件大小
        fhead = struct.pack('128sl', os.path.basename(filepath).encode('utf-8'), os.stat(filepath).st_size)
        # 发送文件名称与文件大小
        s.send(fhead)

        # 将传输文件以二进制的形式分多次上传至服务器
        fp = open(filepath, 'rb')
        while 1:
            data = fp.read(1024)
            if not data:
                print ('{0} file send over...'.format(os.path.basename(filepath)))
                break
            s.send(data)
        # 关闭当期的套接字对象
        s.close()
        
if __name__ == '__main__':
    socket_client()

"""
"""
address = ('192.168.100.22', 2567)
def send(photos):
    for photo in photos[0]:
        print('sending {}'.format(photo))
        data = file_deal(photo)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(address)
        sock.send('{}|{}'.format(len(data), file).encode())    #默认编码 utf-8,发送文件长度和文件名
        reply = sock.recv(1024)
        if 'ok' == reply.decode():             #确认一下服务器get到文件长度和文件名数据
            go = 0
            total = len(data)
            while go < total:                        #发送文件
                data_to_send = data[go:go + 1024]
                sock.send(data_to_send)
                go += len(data_to_send)
            reply = sock.recv(1024)
            if 'copy' == reply.decode():
                print('{} send successfully'.format(photo))
        sock.close()                     #由于tcp是以流的形式传输数据，我们无法判断开头和结尾，简单的方法是没传送一个文件，就使用一个socket，但是这样是消耗计算机的资源，博主正在探索更好的方法，有机会交流一下
        
def file_deal(file_path):    #读取文件的方法
    mes = b''
    try:
        file = open(file_path,'rb')
        mes = file.read()
    except:
        print('error{}'.format(file_path))
    else:
        file.close()
        return mes

        """
"""
        import socket
import os

client = socket.socket()  # 声明socket类型，同时生成socket连接对象
client.connect(('localhost', 6666))  # 链接服务器的ip + 端口

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 获得当前目录

while True:
	msg = input(">>:").strip()  # 获得要向服务端发送的信息，字符串格式
	if len(msg) == 0:
		continue

	client.send(msg.encode("utf-8"))  # 将字符串格式编码成bytes，发送
	if msg == 'break':
		break
	data = client.recv(1024)  # 接收服务端返回的内容
	if len(str(data, 'utf-8').split('|')) == 2:  # 如果返回的字符串长度为2，说明针对的任务2，从服务端传回一张图片
		filename, filesize = str(data, 'utf8').split('|')  # 获得指定图像的名称，图像大小
		path = os.path.join(BASE_DIR, filename)  # 指定图像的保存路径
		filesize = int(filesize)  # 图像大小转换成整形

		f = open(path, 'ab')  # 以二进制格式打开一个文件用于追加。如果该文件不存在，创建新文件进行写入。
		has_receive = 0  # 统计接收到的字节数
		while has_receive != filesize:
			data1 = client.recv(1024)  # 一次从服务端接收1024字节的数据
			f.write(data1)  # 写入
			has_receive += len(data1)  # 更新接收到的字节数
		f.close()  # 关闭文件
	print("recv:", data.decode())

client.close()
"""
