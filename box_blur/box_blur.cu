#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

// prototypes for helper functions
std::string get_image_name(std::string arg);
void init_kernel(float *kernel, int kernel_length);
void image_to_matrix(cv::Mat image, int *blue, int *green, int *red); 
void matrix_to_image(cv::Mat image, int *blue, int *green, int *red);
__global__ void check_pixel(float value);

__global__ void convolve(int *in_blue, int *in_green, int *in_red, int *out_blue, int *out_green, int *out_red,
                            int width, float *kernel, int kernel_length) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int k = kernel_length / 2;
    float blue_sum = 0.0;
    float green_sum = 0.0;
    float red_sum = 0.0;

    for(int i = -k; i <= k; i++) {
        for(int j = -k; j <= k; j++) {
            float kernel_value = kernel[(i + k) * kernel_length + (j + k)];
            int blue_value = in_blue[(i + x) * width + (j + y)];
            int green_value = in_green[(i + x) * width + (j + y)];
            int red_value = in_red[(i + x) * width + (j + y)];

            blue_sum += blue_value * kernel_value;
            green_sum += green_value * kernel_value;
            red_sum += red_value * kernel_value;
        }
    }

    out_blue[x * width + y] = blue_sum;
    out_green[x * width + y] = green_sum;
    out_red[x * width + y] = red_sum;
}


int main(int argc, char **argv) {
    if(argc != 2) {
        std::cout << "usage: ./boxblur <image file to convolve>" << std::endl;
        return 1;
    }
    
    std::string filename = get_image_name(argv[1]);

    if(filename.compare("invalid") == 0) {
        std::cout << "Invalid Image." << std::endl;
        return 1;
    }

    cv::Mat color_image = cv::imread(filename);
    cv::Mat gray_image;
    cv::cvtColor(color_image, gray_image, cv::COLOR_BGR2GRAY);
    int height = gray_image.rows;
    int width = gray_image.cols;

    size_t bytes_image = height * width * sizeof(int);
    int *h_blue = (int*) malloc(bytes_image);
    int *h_green = (int*) malloc(bytes_image);
    int *h_red = (int*) malloc(bytes_image);
    image_to_matrix(color_image, h_blue, h_green, h_red);

    int *d_blue;
    cudaMalloc(&d_blue, bytes_image);
    cudaMemcpy(d_blue, h_blue, bytes_image, cudaMemcpyHostToDevice);
    int *d_green;
    cudaMalloc(&d_green, bytes_image);
    cudaMemcpy(d_green, h_green, bytes_image, cudaMemcpyHostToDevice);
    int *d_red;
    cudaMalloc(&d_red, bytes_image);
    cudaMemcpy(d_red, h_red, bytes_image, cudaMemcpyHostToDevice);

    int kernel_length = 11;
    size_t bytes_kernel = pow(kernel_length, 2) * sizeof(float);
    float *h_kernel = (float*) malloc(bytes_kernel);
    init_kernel(h_kernel, kernel_length); 

    float *d_kernel;
    cudaMalloc(&d_kernel, bytes_kernel);
    cudaMemcpy(d_kernel, h_kernel, bytes_kernel, cudaMemcpyHostToDevice);
    
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
    convolve <<< blocks, threads >>> (d_blue, d_green, d_red, d_result_blue, d_result_green, d_result_red, width, d_kernel, kernel_length);
    long end_time = clock();

    int *h_result_blue = (int*) malloc(bytes_image);
    cudaMemcpy(h_result_blue, d_result_blue, bytes_image, cudaMemcpyDeviceToHost);
    int *h_result_green = (int*) malloc(bytes_image);
    cudaMemcpy(h_result_green, d_result_green, bytes_image, cudaMemcpyDeviceToHost);
    int *h_result_red = (int*) malloc(bytes_image);
    cudaMemcpy(h_result_red, d_result_red, bytes_image, cudaMemcpyDeviceToHost);

    matrix_to_image(color_image, h_result_blue, h_result_green, h_result_red); 
    cv::imwrite("output_cuda.jpg", color_image);

    double conv_time = (double) (end_time - start_time) / CLOCKS_PER_SEC;
    std::cout << "Convolution Time: " << conv_time << "s" << std::endl;

    free(h_blue);
    free(h_green);
    free(h_red);
    free(h_kernel);
    free(h_result_blue);
    free(h_result_green);
    free(h_result_red);
    cudaFree(d_blue);
    cudaFree(d_green);
    cudaFree(d_red);
    cudaFree(d_kernel);
    cudaFree(d_result_blue);
    cudaFree(d_result_green);
    cudaFree(d_result_red);

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
    return "invalid";
}

void init_kernel(float *kernel, int kernel_length) {
    for(int i = 0; i < kernel_length; i++) {
        for(int j = 0; j < kernel_length; j++) {
            kernel[i * kernel_length + j] = 1 / pow(kernel_length, 2);
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

void image_to_matrix(cv::Mat image, int *blue, int *green, int *red) {
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            blue[i * image.cols + j] = image.at<cv::Vec3b>(i, j)[0];
            green[i * image.cols + j] = image.at<cv::Vec3b>(i, j)[1];
            red[i * image.cols + j] = image.at<cv::Vec3b>(i, j)[2];
        }
    }
}

__global__ void check_pixel(float value) {
    if(value > 255)
        value = 255;
    if(value < 0)
        value = 0;
}
