CPP = g++
LD = g++
PRGM = opencv_test

SRC = $(shell find *.cpp)
OBJS = $(SRC: %.cpp=%.o)
FLAGS = -L/usr/local/lib/ -lopencv_core -lopencv_highgui -lopencv_imgproc -I/usr/local/include/opencv4

all: $(PRGM)

$(PRGM): $(OBJS) 
	$(CPP) $(FLAGS) $^ -o $(PRGM)   

$(OBJS): $(SRC) 
	$(LD) $(LD_FLAGS) -c $^

clean: 
	rm -rf *.o $(PRGM)
