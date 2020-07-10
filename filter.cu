#include "image.hpp"

__global__ void invert_color(unsigned char *image, int channels);

void cuda_invert_color(unsigned char *input_image, int height, int width, int channels) {
    unsigned char *gpu_buffer;

    // allocate gpu memory
    cudaMalloc((void**)&gpu_buffer, height * width * channels);

    // copy memory from cpu to gpu
    cudaMemcpy(gpu_buffer, input_image, height * width * channels, cudaMemcpyHostToDevice);

    // perform operation
    dim3 image_matrix(height, width);
    invert_color <<<image_matrix, 1>>> (gpu_buffer, channels);

    // copy processed data back to cpu
    cudaMemcpy(input_image, gpu_buffer, height * width * channels, cudaMemcpyDeviceToHost);

    // free gpu memory
    cudaFree(gpu_buffer);
}

__global__ void invert_color(unsigned char *image, int channels) {
    // for indexing each pixel in image
    int x = blockIdx.x;
    int y = blockIdx.y;
    int index = (x * gridDim.x + y) * channels;

    for(int i = 0; i < channels; i++) {
        image[index + i] = 255 - image[index + 255];
    }
}
