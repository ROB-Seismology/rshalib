#! /bin/bash

swig -python truncated_normal.i

gcc -fPIC -c truncated_normal.c truncated_normal_wrap.c -I/usr/include/python2.7 -Wall -O2 -ansi -pedantic

gcc -shared -o _truncated_normal.so truncated_normal.o truncated_normal_wrap.o -L/usr/lib -lpython2.7

/bin/rm -f *.o
