CC = gcc
LDFLAGS = -shared -pthread -fPIC -fwrapv 
CFLAGS = -O2 -Wall -fno-strict-aliasing \
	-I/usr/include/python2.7 \
	-I/usr/lib/python2.7/site-packages/numpy/core/include/


_data.so: _data.c
	$(CC) $(LDFLAGS) $(CFLAGS)  -o $@ $^

_data.c: _data.pyx
	cython $^

