"C:\Program Files (x86)\pythonxy\swig\swig.exe" -python truncated_normal.i

"C:\MinGW32-xy\bin\mingw32-gcc.exe" -c truncated_normal.c truncated_normal_wrap.c -I"C:\Python27\include" -DBUILD_DLL -Wall -02 -ansi -pedantic

"C:\MinGW32-xy\bin\mingw32-gcc.exe" -shared -o x86\_truncated_normal.pyd truncated_normal.o truncated_normal_wrap.o -L"C:\Python27\libs" -lpython27 -Wl,--out-implib,libtruncated_normal.a

del *.o, *.a