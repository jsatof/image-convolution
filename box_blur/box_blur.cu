#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

// constant gpu memory to hold kernel. (fast access times)
__constant__ double kernel_store[441]; 
int kernel_length = 21;

__global__ void convolve(int *matrix, int height, int width, int kernel_length) {
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int y_index = blockIdx.y * blockDim.y + threadIdx.y;

    double sum = 0;
    int k = kernel_length / 2;

    for(int i = -k; i <= k; i++) {
        for(int j = -k; j <= k; j++) {
            int converted_i = i + k;
            int converted_j = j + k;

            if( converted_i >= 0 && 
                converted_i < kernel_length &&
                converted_j >= 0 &&
                converted_j < kernel_length ) {
                
                    sum += kernel_store[converted_i * kernel_length + converted_j] * (double)matrix[(y_index + j) * width + (x_index + i)];
            }
        }
    }

    matrix[y_index * width + x_index] = sum;
}

void init_kernel() {
    double *host_kernel = (double*) malloc(kernel_length * kernel_length * sizeof(double)); 

    for(int i = 0; i < kernel_length; i++) {
        for(int j = 0; j < kernel_length; j++) {
            host_kernel[i * kernel_length + j] = 1 / kernel_length;
        }
    }

    cudaMemcpy(kernel_store, host_kernel, kernel_length * kernel_length * sizeof(double), cudaMemcpyHostToDevice);

    free(host_kernel);
}

int main(int argc, char **argv) {
    // get image name from command args
    if(argc != 2) {
        printf("usage: ./boxblur <name of image from /images>\n");
        return 1;
    }

    std::string filename;
    if(strcmp(argv[1], "harold") == 0) 
        filename = "../images/harold.jpg";
    else if(strcmp(argv[1], "misha") == 0) 
        filename = "../images/misha_mansoor.jpg";
    else {
        printf("invalid image name\n");
        return 1;
    }

    init_kernel();

    for(int i = 0;)


    cv::Mat image = cv::imread(filename);
    int height = image.rows;
    int width = image.cols;

    int *blue_matrix  = (int*) malloc(height * width * sizeof(int));
    int *green_matrix = (int*) malloc(height * width * sizeof(int));
    int *red_matrix   = (int*) malloc(height * width * sizeof(int));

    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
             blue_matrix[i * width + j] = image.at<cv::Vec3b>(i, j)[0];
            green_matrix[i * width + j] = image.at<cv::Vec3b>(i, j)[1];
              red_matrix[i * width + j] = image.at<cv::Vec3b>(i, j)[2];
        }
    }

    int *device_blue_matrix;
    int *device_green_matrix;
    int *device_red_matrix;

    cudaMalloc((void**)&device_blue_matrix, height * width * sizeof(int));
    cudaMalloc((void**)&device_green_matrix, height * width * sizeof(int));
    cudaMalloc((void**)&device_red_matrix, height * width * sizeof(int));

    cudaMemcpy(device_blue_matrix, blue_matrix, height * width * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_green_matrix, green_matrix, height * width * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_red_matrix, red_matrix, height * width * sizeof(int), cudaMemcpyHostToDevice);

    convolve<<<height, width>>>(device_blue_matrix, height, width, kernel_length);
    convolve<<<height, width>>>(device_green_matrix, height, width, kernel_length);
    convolve<<<height, width>>>(device_red_matrix, height, width, kernel_length);

    cudaMemcpy(blue_matrix, device_blue_matrix, height * width * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(green_matrix, device_green_matrix, height * width * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(red_matrix, device_red_matrix, height * width * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            image.at<cv::Vec3b>(i, j)[0] =  blue_matrix[i * width + j];
            image.at<cv::Vec3b>(i, j)[1] = green_matrix[i * width + j];
            image.at<cv::Vec3b>(i, j)[2] =   red_matrix[i * width + j];
        }
    }

    cv::imwrite("harold.jpg", image);

    cudaFree(device_blue_matrix);
    cudaFree(device_green_matrix);
    cudaFree(device_red_matrix);

    free(blue_matrix);
    free(green_matrix);
    free(red_matrix);

    return 0;
}