llvm-prof
===========

this was copied form llvm release 3.3 include llvm-prof libLLVMProfiling
libprofile\_rt.so 

because since llvm release 3.4 they remove profiling related code totaly, so
this project would help you get the function back.

build
------

	$ mkdir build
	$ cd build
	$ cmake ..
	$ make 
	$ sudo make install

note
-----

you need add `$PKG_CONFIG_PATH` to let pkg-config recognise llvm-prof.pc