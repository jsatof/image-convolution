NVCC = nvcc
NV_FLAGS = -g -L/lib -lc++

PRGM = run
SRCS = $(shell find *.cu) $(shell find *.cpp)
OBJS = image.o filter.o
HEAD = image.hpp
OPENCV = -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs 

$PRGM: $(OBJS)
	$(NVCC) $(NV_FLAGS) -o run $^ $(OPENCV)

$(OBJS): $(SRCS)
	$(NVCC) $(NV_FLAGS) -c $^ $(OPENCV) 

clean: 
	rm -rf *.o $(PRGM)
