#https://blog.csdn.net/m0_38106923/article/details/86562451 树莓派/PC实现实时摄像头数据共享—最优方法（搭建网络摄像头）
#https://blog.csdn.net/u012736685/article/details/77131633  c++ socket
import numpy as np
import cv2
import socket
class VideoStreamingTest(object):
    def __init__(self, host, port):
 
        self.server_socket = socket.socket()
        self.server_socket.bind((host, port))
        self.server_socket.listen(0)
        self.connection, self.client_address = self.server_socket.accept()
        self.connection = self.connection.makefile('rb')
        #self.host_name = socket.gethostname()
        #self.host_ip = socket.gethostbyname(self.host_name)
        self.streaming()
 
    def streaming(self):
 
        try:
            #print("Host: ", self.host_name + ' ' + self.host_ip)
            print("Connection from: ", self.client_address)
            print("Streaming...")
            print("Press 'q' to exit")
 
            # need bytes here
            stream_bytes = b' '
            while True:
                stream_bytes += self.connection.read(1024)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    cv2.imshow('image', image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            self.connection.close()
            self.server_socket.close()
 
 
if __name__ == '__main__':
    # host, port
    h, p = "192.168.0.100", 8000
    VideoStreamingTest(h, p)


"""
#!coding=utf8
import socket
import threading
import struct
import time
import cv2
import numpy
 
class Carame_Accept_Object:
    def __init__(self,S_addr_port=("192.168.0.100",8881)):
        self.resolution=(640,480)       #分辨率
        self.img_fps=15                 #每秒传输多少帧数
        self.addr_port=S_addr_port
        self.Set_Socket(self.addr_port)
 
    #设置套接字
    def Set_Socket(self,S_addr_port):
        self.server=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1) #端口可复用
        self.server.bind(S_addr_port)
        self.server.listen(5)
        print("the process work in the port:%s" % S_addr_port[0])
 
 
def check_option(object,client):
    #按格式解码，确定帧数和分辨率
    info=struct.unpack('lhh',client.recv(8))
    if info[0]>888:
        object.img_fps=int(info[0])-888          #获取帧数
        object.resolution=list(object.resolution)
        # 获取分辨率
        object.resolution[0]=info[1]
        object.resolution[1]=info[2]
        object.resolution = tuple(object.resolution)
        return 1
    else:
        return 0
 
def RT_Image(object,client,D_addr):
    if(check_option(object,client)==0):
        return
    camera=cv2.VideoCapture(0)                                #从摄像头中获取视频
    img_param=[int(cv2.IMWRITE_JPEG_QUALITY),object.img_fps]  #设置传送图像格式、帧数
    while(1):
        time.sleep(0.1)             #推迟线程运行0.1s
        _,object.img=camera.read()  #读取视频每一帧
 
        object.img=cv2.resize(object.img,object.resolution)     #按要求调整图像大小(resolution必须为元组)
        _,img_encode=cv2.imencode('.jpg',object.img,img_param)  #按格式生成图片
        img_code=numpy.array(img_encode)                        #转换成矩阵
        object.img_data=img_code.tostring()                     #生成相应的字符串
        try:
            #按照相应的格式进行打包发送图片
            client.send(struct.pack("lhh",len(object.img_data),object.resolution[0],object.resolution[1])+object.img_data)
        except:
            camera.release()        #释放资源
            return
 
if __name__ == '__main__':
    camera=Carame_Accept_Object()
    while(1):
        client,D_addr=camera.server.accept()
        clientThread=threading.Thread(None,target=RT_Image,args=(camera,client,D_addr,))
        clientThread.start()
"""
"""
#!coding=utf-8

import threading
import socket
import struct
import sys
def socket_service():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 绑定端口为9001
        s.bind(('127.0.0.1', 9001))
        # 设置监听数
        s.listen(10)
    except socket.error as msg:
        print (msg)
        sys.exit(1)
    print ('Waiting connection...')
 
    while 1:
        # 等待请求并接受(程序会停留在这一旦收到连接请求即开启接受数据的线程)
        conn, addr = s.accept()#当有请求到指定端口是 accpte()会返回一个新的socket和对方主机的（ip,port）
        # 接收数据
        t = threading.Thread(target=deal_data, args=(conn, addr))
        t.start()
 
def deal_data(conn, addr):
    print ('Accept new connection from {0}'.format(addr))
    # conn.settimeout(500)
    # 收到请求后的回复
    conn.send('Hi, Welcome to the server!'.encode('utf-8'))
 
    while 1:
        # 申请相同大小的空间存放发送过来的文件名与文件大小信息
        fileinfo_size = struct.calcsize('128sl')
        # 接收文件名与文件大小信息
        buf = conn.recv(fileinfo_size)
        # 判断是否接收到文件头信息
        if buf:
            # 获取文件名和文件大小
            filename, filesize = struct.unpack('128sl', buf)
            fn = filename.strip(b'\00')
            fn = fn.decode()
            print ('file new name is {0}, filesize if {1}'.format(str(fn),filesize))
 
            recvd_size = 0  # 定义已接收文件的大小
            # 存储在该脚本所在目录下面
            fp = open('./' + str(fn), 'wb')
            print ('start receiving...')
            
            # 将分批次传输的二进制流依次写入到文件
            while not recvd_size == filesize:
                if filesize - recvd_size > 1024:
                    data = conn.recv(1024)
                    recvd_size += len(data)
                else:
                    data = conn.recv(filesize - recvd_size)
                    recvd_size = filesize
                fp.write(data)
            fp.close()
            print ('end receive...')
        # 传输结束断开连接
        conn.close()
        break
        
if __name__ == "__main__":
    socket_service()
"""
"""

LOCAL_IP = '192.168.100.22'   #本机在局域网中的地址，或者写127.0.0.1
PORT = 2567                   #指定一个端口
def server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # socket.AF_INET 指ipv4  socket.SOCK_STREAM 使用tcp协议
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) #设置端口
    sock.bind((LOCAL_IP, PORT))       #绑定端口
    sock.listen(3)                    #监听端口
    while True:
        sc,sc_name = sock.accept()    #当有请求到指定端口是 accpte()会返回一个新的socket和对方主机的（ip,port）
        print('收到{}请求'.format(sc_name))
        infor = sc.recv(1024)       #首先接收一段数据，这段数据包含文件的长度和文件的名字，使用|分隔，具体规则可以在客户端自己指定
        length,file_name = infor.decode().split('|')
        if length and file_name:
            newfile = open('image/'+str(random.randint(1,10000))+'.jpg','wb')  #这里可以使用从客户端解析出来的文件名
            print('length {},filename {}'.format(length,file_name))
            sc.send(b'ok')   #表示收到文件长度和文件名
            file = b''
            total = int(length)
            get = 0
            while get < total:         #接收文件
                data = sc.recv(1024)
                file += data
                get = get + len(data)
            print('应该接收{},实际接收{}'.format(length,len(file)))
            if file:
                print('acturally length:{}'.format(len(file)))
                newfile.write(file[:])
                newfile.close()
                sc.send(b'copy')    #告诉完整的收到文件了
        sc.close()
"""#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Date: 19-5-22 下午8:48

"""
import socket
import os

server = socket.socket()  # 1.声明协议类型，同时生成socket链接对象
server.bind(('localhost', 6666))  # 绑定要监听端口=(服务器的ip地址+任意一个端口)
server.listen(5)  # 监听

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 获得当前目录
PIC_DIR = os.path.join(BASE_DIR, 'pics')  # 图片文件夹的路径
pic_names = sorted(os.listdir(PIC_DIR))  # 获得排序后的所有图片名称，放在一个列表中【'1.jpg','2.jpg',...,'9.jpg'】
# 将图片名称放进一个大字符串中
pic_names = " ".join(i for i in pic_names)  # '0.jpg 1.jpg 2.jpg 3.jpg 4.jpg 5.jpg 6.jpg 7.jpg 8.jpg 9.jpg'

print("我要开始等电话了")

while True:
	conn, addr = server.accept()  # 等电话打进来
	# conn就是客户端连过来而在服务器端为其生成的一个连接实例
	print("收到来自{}请求".format(addr))

	while True:
		data = conn.recv(1024)  # 接收数据，获取图像名称的命令，指定需要传输的图片
		print("recv:", data)

		if not data:
			print("client has lost...")
			break
		if data.decode('utf-8') == 'get pics_names':  # 获取图像名称的命令  定义为get pics_names
			conn.send(pic_names.encode('utf-8'))
		elif data.decode('utf-8') == 'break':
			break
		else:
			img_name = data.decode('utf-8')  # 将客户端传输过来的图片名称（bytes）解码成字符串
			img_path = os.path.join(PIC_DIR, img_name)  # 获得对应图片的绝对路径
			file_size = os.stat(img_path).st_size  # 获得图像文件的大小，字节单位
			file_info = '%s|%s' % (img_name, file_size)  # 将图像信息发给客户端（用于区分文档中的两个任务）
			conn.sendall(bytes(file_info, 'utf-8'))

			f = open(img_path, 'rb')  # 以二进制格式打开一个文件用于只读
			has_sent = 0  # 记录下已经发送的字节数
			while has_sent != file_size:  # 发送的字节数 不等于 图像的大小，则接着发送
				file = f.read(1024)  # 一次读1024个字节
				conn.sendall(file)  # 发送给客户端
				has_sent += len(file)  # 更新已发送的字节数
			f.close()  # 发送结束，关闭文件
			print('上传成功')
	break
"""

