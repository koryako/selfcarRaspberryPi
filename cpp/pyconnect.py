import ctypes
so = ctypes.cdll.LoadLibrary
lib = so("../build/libclient.dylib")

lib.go(1,'127.0.0.1')