NVCC = nvcc
NV_FLAGS = -g -L/lib -lc++

PRGM = run
SRCS = $(shell find *.cu) 
OBJS = $(SRCS:*.cu=*.o)
HEAD = cuhelper.cu
OPENCV = -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc

$(PRGM): $(OBJS)
	$(NVCC) $(NV_FLAGS) -o run $^ $(HEAD) $(OPENCV) -lm

$(OBJS): $(SRCS)
	$(NVCC) $(NV_FLAGS) -c $^ $(HEAD) $(OPENCV) -lm

serial: serial.cpp
	g++ $(NV_FLAGS) -o serial serial.cpp $(OPENCV)

clean: 
	rm -rf *.o $(PRGM)
