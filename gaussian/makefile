NVCC = nvcc
NV_FLAGS = -g -L/lib -lc++ -I/usr/local/include/opencv4 -L/usr/local/lib

PRGM = gaussian
SRCS = gaussian.cu
OBJS = gaussian.o
LIBS = -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lc++ -lm

$(PRGM): $(OBJS)
	$(NVCC) $(NV_FLAGS) -o $@ $^ $(LIBS)

$(OBJS): $(SRCS)
	$(NVCC) $(NV_FLAGS) -c $^ $(HEAD) $(LIBS) 

serial: serial.cpp
	g++ $(NV_FLAGS) -o serial serial.cpp  $(LIBS)

clean: 
	rm -rf *.o $(PRGM) serial
