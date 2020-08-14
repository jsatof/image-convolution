# Cuda Image Convolution
My implementations of a variety of kernels on images. It demonstrates what I have gathered in my endeavours in parallel GPU computing with the Nvidia CUDA toolkit. 

This project is run on my Linux machine. I have not programmed with portability in mind, so results on other operating systems are not guarenteed.

Each folder contains a serial and CUDA implementation of the filter. Some are WIP.

## Dependencies
- CUDA Toolkit 10.1
- Nvidia Drivers v440xx
- OpenCV
- libstdc++ (conflict with old c++ std::strings)

## System Specs
- Nvidia GTX 960M
- Intel i7-6700HQ @ 2.3GHz
- 16GB RAM
- OS Manjaro Linux 20.0.3
