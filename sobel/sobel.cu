#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

int sobel_x[9] = {
    -1, 0, 1,
    -2, 0, 1,
    -1, 0, 1
};

int sobel_y[9] = {
    1, 2, 1,
    0, 0, 0,
    -1, -2, -1
};
int kernel_length = 3;
int k = 1;


__global__ void convolve(int *matrix, int *kernel, int height, int width) {
    int index= blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0;

    for(int i = -1; i <= 1; i++) {
        for(int j = -1; j <= 1; j++) {
               
            sum += kernel[(i + 1) * 3 + (j + 1)] * matrix[index];
        }
    }

    matrix[index] = sum;
}

// convolving with sobel kernels involve a pythagorean calculation
__global__ void combine_convolved(int *a, int *b, int *c) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int aa = a[index];
    int bb = b[index];

    c[index] = (int) sqrtf(aa * aa + bb * bb);
}

int main(int argc, char **argv) {
    if(argc != 2) {
        printf("usage: ./run <name of image>\n");
        return 1;
    }

    std::string filename;
    if(strcmp(argv[1], "harold") == 0) {
        filename = "../images/harold.jpg";
    } else {
        printf("Invalid image name\n");
        return 1;
    }

    cv::Mat image = cv::imread(filename);
    int height = image.rows;
    int width = image.cols;

    int *matrix_x = (int*) malloc(height * width * sizeof(int));
    int *matrix_y = (int*) malloc(height * width * sizeof(int));

    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            matrix_x[i * width + j] = image.at<uchar>(i, j);
            matrix_y[i * width + j] = image.at<uchar>(i, j);
        }
    }

    int *device_gradient_x;
    int *device_gradient_y;
    int *combined_matrix;
    cudaMalloc(&device_gradient_x, height * width * sizeof(int));
    cudaMalloc(&device_gradient_y, height * width * sizeof(int));
    cudaMalloc(&combined_matrix, height * width * sizeof(int));

    convolve<<< height, width >>>(device_gradient_x, sobel_x, height, width);
    convolve<<< height, width >>>(device_gradient_y, sobel_y, height, width);
    combine_convolved<<< height, width>>>(device_gradient_x, device_gradient_y, combined_matrix);

    int *result_matrix = (int*) malloc(height * width * sizeof(int));
    cudaMemcpy(result_matrix, combined_matrix, height * width * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            image.at<uchar>(i, j) = result_matrix[i * width + j];
        }
    }

    cv::imwrite("harold.jpg", image);
    
    return 0;
}