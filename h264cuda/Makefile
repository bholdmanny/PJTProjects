all: src cuda
	nvcc -I ~/NVIDIA_GPU_Computing_SDK/C/common/inc/ -g -G -arch=sm_13 -o cuve264b cuve264b_main.cpp  cuda/encode_cuda.o src/*.o
	mv cuve264b bin/

src: ALWAYS
	(cd src; make)

cuda: ALWAYS
	(cd cuda; make)


ALWAYS:

clean:
	rm src/*.o
	rm cuda/*.o
	rm bin/cuve264b
