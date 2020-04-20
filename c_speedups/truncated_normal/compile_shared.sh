#! /bin/bash

gcc -fPIC -c -O3 -Wall -Wno-long-long -ansi -pedantic -D_POSIX_SOURCE *.c
gcc -shared -o libtruncated_normal.so *.o
/bin/rm -f *.o
