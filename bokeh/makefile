NVCC = nvcc
NV_FLAGS = -g -L/lib -lc++

PRGM = bokeh
SRCS = $(shell find *.cu) 
OBJS = $(SRCS:*.cu=*.o)

OPENCV = -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc

$(PRGM): $(OBJS)
	$(NVCC) $(NV_FLAGS) -o run $^ $(OPENCV) -lm

$(OBJS): $(SRCS)
	$(NVCC) $(NV_FLAGS) -c $^ $(OPENCV) -lm

serial: serial.cpp
	g++ -g -L/lib -lc++ -o serial serial.cpp  $(OPENCV)

clean: 
	rm -rf *.o $(PRGM) serial
