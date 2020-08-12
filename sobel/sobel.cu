#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

std::string get_image_name(std::string arg);
void image_to_matrix(cv::Mat image, int *matrix);
void matrix_to_image(cv::Mat image, int *matrix);

__global__ void convolve(int *input, int *output, int width, double *kernel, int kernel_length) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int k = kernel_length / 2;
    double sum = 0.0;

    for(int i = -k; i <= k; i++) {
        for(int j = -k; j <= k; j++) {
            double kernel_value = kernel[(i + k) * kernel_length + (j + k)];
            int matrix_value = input[(i + x) * width + (j + y)];

            sum += matrix_value * kernel_value;
        }
    }

    output[x * width + y] = sum;
}

// convolving with sobel kernels involve a pythagorean calculation
__global__ void get_magnitude(int *x_matrix, int *y_matrix, int *out_matrix, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int aa = x_matrix[x * width + y];
    int bb = y_matrix[x * width + y];

    out_matrix[x * width + y] = (int) sqrtf(aa * aa + bb * bb);
}

int main(int argc, char **argv) {
    if(argc != 2) {
        printf("usage: ./boxblur <image file to convolve>\n");
        return 1;
    }
    
    std::string filename = get_image_name(argv[1]);

    if(filename.compare("invalid") == 0) {
        printf("Invalid Image.\n");
        return 1;
    }

    cv::Mat color_image = cv::imread(filename);
    cv::Mat gray_image;
    cv::cvtColor(color_image, gray_image, cv::COLOR_BGR2GRAY);
    int height = gray_image.rows;
    int width = gray_image.cols;

    size_t bytes_image = height * width * sizeof(int);
    int *h_matrix_x = (int*) malloc(bytes_image);
    int *h_matrix_y = (int*) malloc(bytes_image);
    image_to_matrix(gray_image, h_matrix_x);
    image_to_matrix(gray_image, h_matrix_y);

    int *d_matrix_x;
    cudaMalloc(&d_matrix_x, bytes_image);
    cudaMemcpy(d_matrix_x, h_matrix_x, bytes_image, cudaMemcpyHostToDevice);
    int *d_matrix_y;
    cudaMalloc(&d_matrix_y, bytes_image);
    cudaMemcpy(d_matrix_y, h_matrix_y, bytes_image, cudaMemcpyHostToDevice);

    int kernel_length = 5;
    // these 5x5 sobel kernels are not my discovery. 
    // courtesy of https://stackoverflow.com/questions/9567882/sobel-filter-kernel-of-large-size/41065243#41065243
    double h_sobel_x[25] = {
        -2/8, -1/5,  0,  1/5,  2/8,
        -2/5, -1/2,  0,  1/2,  2/5,
        -2/4, -1/1,  0,  1/1,  2/4,
        -2/5, -1/2,  0,  1/2,  2/5,
        -2/8, -1/5,  0,  1/5,  2/8
    };

    double h_sobel_y[25] = {
        -2/8, -2/5, -2/4, -2/5, -2/8,
        -1/5, -1/2, -1/1, -1/2, -1/5,
        0,    0,    0,    0,    0,
        1/5, 1/2, 1/1, 1/2, 1/5,
        -2/8, -2/5, -2/4, -2/5, -2/8,
    };
    /*
    int kernel_length = 3;
    double h_sobel_x[9] = {
        -1.0, 0.0, 1.0,
        -2.0, 0.0, 2.0,
        -1.0, 0.0, 1.0,
    };

    const float h_sobel_y[9] = {
        1.0, 2.0, 1.0,
        0.0, 0.0, 0.0,
        -1.0, -2.0, -1.0,
    };
    */

    double *d_sobel_x;
    cudaMalloc(&d_sobel_x, sizeof(h_sobel_x));
    cudaMemcpy(d_sobel_x, h_sobel_x, sizeof(h_sobel_x), cudaMemcpyHostToDevice);
    double *d_sobel_y;
    cudaMalloc(&d_sobel_y, sizeof(h_sobel_y));
    cudaMemcpy(d_sobel_y, h_sobel_y, sizeof(h_sobel_y), cudaMemcpyHostToDevice);

    int *d_result_matrix_x;
    cudaMalloc(&d_result_matrix_x, bytes_image);
    int *d_result_matrix_y;
    cudaMalloc(&d_result_matrix_y, bytes_image);
    int *d_combined_result;
    cudaMalloc(&d_combined_result, bytes_image);

    int num_threads = 16;
    dim3 threads(num_threads, num_threads);
    dim3 blocks(width / num_threads + 1, height / num_threads + 1);

    long start_time = clock();
    convolve <<< blocks, threads >>> (d_matrix_x, d_result_matrix_x, width, d_sobel_x, kernel_length);
    convolve <<< blocks, threads >>> (d_matrix_y, d_result_matrix_y, width, d_sobel_y, kernel_length);
    get_magnitude <<< blocks, threads >>> (d_result_matrix_x, d_result_matrix_y, d_combined_result, width);
    long end_time = clock();

    int *h_combined_result = (int*) malloc(bytes_image);
    cudaMemcpy(h_combined_result, d_combined_result, bytes_image, cudaMemcpyDeviceToHost);

    matrix_to_image(gray_image, h_combined_result);
    cv::imwrite("output_cuda.jpg", gray_image);

    double conv_time = (double) (end_time - start_time) / CLOCKS_PER_SEC;
    printf("Convolution Time: %fs\n", conv_time);

    free(h_combined_result);
    free(h_matrix_x);
    free(h_matrix_y);
    cudaFree(d_combined_result);
    cudaFree(d_matrix_x);
    cudaFree(d_matrix_y);
    cudaFree(d_result_matrix_x);
    cudaFree(d_result_matrix_y);
    cudaFree(d_sobel_x);
    cudaFree(d_sobel_y);

    return 0;
}

std::string get_image_name(std::string arg) {
    if(arg.compare("harold.jpg") == 0 || arg.compare("harold") == 0) {
        return "../images/harold.jpg";
    } 
    if(arg.compare("misha_mansoor.jpg") == 0 || arg.compare("misha") == 0) {
        return "../images/misha_mansoor.jpg";
    }
    if(arg.compare("christmas.jpg") == 0 || arg.compare("xmas") == 0) {
        return "../images/christmas.jpg";
    }
    if(arg.compare("nier.jpg") == 0 || arg.compare("nier") == 0) {
        return "../images/nier.jpg";
    }
    if(arg.compare("band.jpg") == 0 || arg.compare("band") == 0) {
        return "../images/band.jpg";
    }
    return "invalid";
}

void image_to_matrix(cv::Mat image, int *matrix) {
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            matrix[i * image.cols + j] = image.at<uchar>(i, j);
        }
    }
}

void matrix_to_image(cv::Mat image, int *matrix) {
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            image.at<uchar>(i, j) = matrix[i * image.cols + j];
        }
    }
}
