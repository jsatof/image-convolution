#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <opencv2/opencv.hpp>

std::string get_image_name(std::string arg);
void image_to_matrix(cv::Mat image, int *blue, int *green, int *red);
void matrix_to_image(cv::Mat image, int *blue, int *green, int *red);

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

int main(int argc, char **argv) {
    if(argc != 2) {
        printf("usage: ./run <name of image file>\n");
        return 1;
    }

    std::string filename = get_image_name(argv[1]);

    if(filename.compare("invalid") == 0) {
        printf("Invalid Image.\n");
        return 1;
    }

    cv::Mat image = cv::imread(filename);
    int height = image.rows;
    int width = image.cols;

    size_t bytes_image = height * width * sizeof(int);
    int *h_blue = (int*) malloc(bytes_image);
    int *h_green = (int*) malloc(bytes_image);
    int *h_red = (int*) malloc(bytes_image);
    image_to_matrix(image, h_blue, h_green, h_red);

    int kernel_length = 3;
    double h_kernel[9] = {
        0.0, -1.0, 0.0,
        -1.0, 5.0, -1.0,
        0.0, -1.0, 0.0
    };

    double *d_kernel;
    cudaMalloc(&d_kernel, sizeof(h_kernel));
    cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

    int *d_blue;
    cudaMalloc(&d_blue, bytes_image);
    cudaMemcpy(d_blue, h_blue, bytes_image, cudaMemcpyHostToDevice);
    int *d_green;
    cudaMalloc(&d_green, bytes_image);
    cudaMemcpy(d_green, h_green, bytes_image, cudaMemcpyHostToDevice);
    int *d_red;
    cudaMalloc(&d_red, bytes_image);
    cudaMemcpy(d_red, h_red, bytes_image, cudaMemcpyHostToDevice);

    int *d_result_blue;
    cudaMalloc(&d_result_blue, bytes_image);
    int *d_result_green;
    cudaMalloc(&d_result_green, bytes_image);
    int *d_result_red;
    cudaMalloc(&d_result_red, bytes_image);

    int num_threads = 16;
    dim3 threads(num_threads, num_threads);
    dim3 blocks(width / num_threads + 1, height / num_threads + 1);

    long start_time = clock();
    convolve <<< blocks, threads >>> (d_blue, d_result_blue, width, d_kernel, kernel_length);
    convolve <<< blocks, threads >>> (d_green, d_result_green, width, d_kernel, kernel_length);
    convolve <<< blocks, threads >>> (d_red, d_result_red, width, d_kernel, kernel_length);
    long end_time = clock();

    int *h_result_blue = (int*) malloc(bytes_image);
    cudaMemcpy(h_result_blue, d_result_blue, bytes_image, cudaMemcpyDeviceToHost);
    int *h_result_green = (int*) malloc(bytes_image);
    cudaMemcpy(h_result_green, d_result_green, bytes_image, cudaMemcpyDeviceToHost);
    int *h_result_red = (int*) malloc(bytes_image);
    cudaMemcpy(h_result_red, d_result_red, bytes_image, cudaMemcpyDeviceToHost);
 
    matrix_to_image(image, h_result_blue, h_result_green, h_result_red);
    cv::imwrite("output_cuda.jpg", image);

    double conv_time = (double) (end_time - start_time) / CLOCKS_PER_SEC;
    printf("Convolution Time: %fs\n", conv_time);

    free(h_blue);
    free(h_green);
    free(h_red);
    free(h_result_blue);
    free(h_result_green);
    free(h_result_red);
    cudaFree(d_blue);
    cudaFree(d_green);
    cudaFree(d_red);
    cudaFree(d_result_blue);
    cudaFree(d_result_green);
    cudaFree(d_result_red);
    cudaFree(d_kernel);

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
    return "invalid";
}

void image_to_matrix(cv::Mat image, int *blue, int *green, int *red) {
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            blue[i * image.cols + j] = image.at<cv::Vec3b>(i, j)[0];
            green[i * image.cols + j] = image.at<cv::Vec3b>(i, j)[1];
            red[i * image.cols + j] = image.at<cv::Vec3b>(i, j)[2];
        }
    }
}

void matrix_to_image(cv::Mat image, int *blue, int *green, int *red) {
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            image.at<cv::Vec3b>(i, j)[0] = blue[i * image.cols + j];
            image.at<cv::Vec3b>(i, j)[1] = green[i * image.cols + j];
            image.at<cv::Vec3b>(i, j)[2] = red[i * image.cols + j];
        }
    }
}