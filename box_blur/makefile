NVCC = nvcc
NV_FLAGS = -g -L/lib -lc++

PRGM = run
SRCS = $(shell find *.cu) 
OBJS = $(SRCS:*.cu=*.o)
HEAD = image.hpp
OPENCV = -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs 

$(PRGM): $(OBJS)
	$(NVCC) $(NV_FLAGS) -o run $^ $(OPENCV) -lm

$(OBJS): $(SRCS)
	$(NVCC) $(NV_FLAGS) -c $^ $(OPENCV) -lm

clean: 
	rm -rf *.o $(PRGM)
