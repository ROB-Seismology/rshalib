swig -python truncated_normal.i

gcc -DMS_WIN64 -mdll -O2 -Wall -IC:\Miniconda3\envs\py2\include -IC:\Miniconda3\envs\py2\Lib\site-packages\numpy\core\include -LC:\Miniconda3\envs\py2\libs truncated_normal.c truncated_normal_wrap.c -lpython27 -o _truncated_normal.pyd

In Mingw64 environment:
gcc -DMS_WIN64 -mdll -O2 -Wall -I/C/Miniconda3/envs/py3/include -I/C/Miniconda3/envs/py3/Lib/site-packages/numpy/core/include -L/C/Miniconda3/envs/py3 truncated_normal.c truncated_normal_wrap.c -lpython37 -o _truncated_normal.cp37-win_amd64.pyd