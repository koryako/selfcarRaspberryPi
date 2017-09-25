import ctypes
so = ctypes.cdll.LoadLibrary
lib = so("../build/libserver.dylib")

print (lib.main())
"""
/***gcc -o libpycall.so -shared -fPIC pycall.c*/ 
#include <stdio.h> 
#include <stdlib.h> 
int foo(int a, int b) 
{ 
  printf("you input %d and %d\n", a, b); 
  return a+b; 
} 

import ctypes 
ll = ctypes.cdll.LoadLibrary  
lib = ll("./libpycall.so")   
lib.foo(1, 3) 
print '***finish***' 
 
"""