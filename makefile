CPP = g++
CPP_FLAGS = -g -Wall
LD = g++
LD_FLAGS = -g -Wall
PRGM = opencv_test

SRCS = $(shell find *.cpp)
OBJS = $(SRCS:%.o=%.cpp)
OPENCV = -I/usr/local/include/opencv4/ -L/usr/local/lib/ -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs

$(PRGM): $(OBJS)
	$(CPP) $(CPP_FLAGS) $^ -o $(PRGM) $(OPENCV)

$(OBJS): $(SRCS) 
	$(LD) $(LD_FLAGS) -c $^ $(OPENCV)

clean: 
	rm -rf *.o $(PRGM)
