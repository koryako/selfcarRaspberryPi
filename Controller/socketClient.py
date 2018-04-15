from socket import *
import time


HOST="192.168.0.108"
PORT=8888

s=socket(AF_INET,SOCK_STREAM)
s.bind((HOST,PORT))
s.listen(1)
print ("listening on 8888")

while 1:
    conn,addr=s.accept()
    print ("Connnected by:",addr)
    while 1:
           command=conn.recv(1024).replace('\n','')
           if not command:
                break
           print command
           conn.send('33') 
           #execute(command)
    conn.close()





    
    
        



