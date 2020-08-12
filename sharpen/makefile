NVCC = nvcc
NV_FLAGS = -g -L/lib -lc++ -I/usr/local/include/opencv4 -L/usr/local/lib

PRGM = sharpen
SRCS = sharpen.cu
OBJS = sharpen.o
LIBS = -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lstdc++ -lm

$(PRGM): $(OBJS)
	$(NVCC) $(NV_FLAGS) -o $@ $^ $(LIBS)

$(OBJS): $(SRCS)
	$(NVCC) $(NV_FLAGS) -c $^ 

serial: serial.cpp
	g++ $(NV_FLAGS) -o serial serial.cpp  $(LIBS)

clean: 
	rm -rf *.o $(PRGM) serial
