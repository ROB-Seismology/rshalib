#!/bin/bash
#
cp truncated_normal.h /$HOME/include
#
gcc -c -g -I/$HOME/include truncated_normal.c >& compiler.txt
if [ $? -ne 0 ]; then
  echo "Errors compiling truncated_normal.c"
  exit
fi
rm compiler.txt
#
mv truncated_normal.o ~/libc/$ARCH/truncated_normal.o
#
echo "Library installed as ~/libc/$ARCH/truncated_normal.o"
