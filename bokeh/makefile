NVCC = nvcc
NV_FLAGS = -g  -I/usr/local/include/opencv4 

PRGM = bokeh
SRCS = $(shell find *.cu) 
OBJS = $(SRCS:*.cu=*.o)

LIBS = -L/lib -L/usr/local/lib -lstdc++ -lm -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc

$(PRGM): $(OBJS)
	$(NVCC) $(NV_FLAGS) -o $@ $^ $(LIBS)

$(OBJS): $(SRCS)
	$(NVCC) $(NV_FLAGS) -c $^ 

serial: serial.cpp
	g++ $(NV_FLAGS) -o serial serial.cpp  $(LIBS)

clean: 
	rm -rf *.o $(PRGM) serial
