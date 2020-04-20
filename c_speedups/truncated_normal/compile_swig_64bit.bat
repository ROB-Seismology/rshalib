swig -python truncated_normal.i

"C:\Anaconda\Scripts\gcc.bat" -DMS_WIN64 -mdll -O2 -Wall -IC:\Anaconda\include -IC:\Anaconda\Lib\site-packages\numpy\core\include -LC:\Anaconda\libs truncated_normal.c truncated_normal_wrap.c -lpython27 -o _truncated_normal.pyd

In Mingw64 environment:
gcc -DMS_WIN64 -mdll -O2 -Wall -I/C/Anaconda/envs/py3/include -I/C/Anaconda/env s/py3/Lib/site-packages/numpy/core/include -L/C/Anaconda/envs/py3 truncated_normal.c truncated_norm al_wrap.c -lpython36 -o _truncated_normal.pyd