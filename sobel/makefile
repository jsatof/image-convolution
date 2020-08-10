NVCC = nvcc
NV_FLAGS = -g -L/lib -I/usr/local/include/opencv4 -L/usr/local/lib

PRGM = sobel
SRCS = sobel.cu
OBJS = sobel.o
LIBS = -lc++ -lm -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc

$(PRGM): $(OBJS)
	$(NVCC) $(NV_FLAGS) -o $@ $^ $(LIBS)

$(OBJS): $(SRCS)
	$(NVCC) $(NV_FLAGS) -c $^

serial: serial.cpp
	g++ $(NV_FLAGS) -o serial serial.cpp $(LIBS)

clean: 
	rm -rf *.o $(PRGM)
