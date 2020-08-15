#include <stdio.h>
#include <malloc.h>
#include <time.h>
#include <opencv2/opencv.hpp>

std::string get_image_name(std::string arg);
void image_to_matrix(cv::Mat image, int *blue, int *green, int *red);
void matrix_to_image(cv::Mat image, int *blue, int *green, int *red);
int inside_circle(int x, int y, int r, int k);
void init_kernel(double *kernel, int kernel_length);

__global__ void convolve(int *input, int *output, int width, double *kernel, int kernel_length) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int k = kernel_length / 2;

    double sum = 0;
    for(int i = -k; i <= k; i++) {
        for(int j = -k; j <= k; j++) {
            sum += kernel[(i + k) * kernel_length + (j + k)] * input[(x + i) * width + (y + j)];
        }
    }

    output[x * width + y] = (int) sum;
}

int main(int argc, char **argv) {
    if(argc != 2) {
        printf("usage: ./run <name of image>\n");
        return 1;
    }

    std::string filename = get_image_name(argv[1]);

    if(!filename.compare("invalid")) {
        printf("invalid image\n");
        return 1;
    }

    cv::Mat image = cv::imread(filename);
    int height = image.rows;
    int width = image.cols;
    size_t bytes_matrix = height * width * sizeof(int);
    
    int *h_blue = (int*) malloc(bytes_matrix);
    int *h_green = (int*) malloc(bytes_matrix);
    int *h_red = (int*) malloc(bytes_matrix);
    image_to_matrix(image, h_blue, h_green, h_red);

    int *d_blue;
    cudaMalloc(&d_blue, bytes_matrix);
    cudaMemcpy(d_blue, h_blue, bytes_matrix, cudaMemcpyHostToDevice);
    int *d_green;
    cudaMalloc(&d_green, bytes_matrix);
    cudaMemcpy(d_green, h_green, bytes_matrix, cudaMemcpyHostToDevice);
    int *d_red;
    cudaMalloc(&d_red, bytes_matrix);
    cudaMemcpy(d_red, h_red, bytes_matrix, cudaMemcpyHostToDevice);

    int kernel_length = 10;
    size_t bytes_kernel = pow(kernel_length, 2) * sizeof(double);

    double *h_kernel = (double*) malloc(bytes_kernel);
    init_kernel(h_kernel, kernel_length);
    
    double *d_kernel;
    cudaMalloc(&d_kernel, bytes_kernel);
    cudaMemcpy(d_kernel, h_kernel, bytes_kernel, cudaMemcpyHostToDevice);

    int *d_blue_result;
    cudaMalloc(&d_blue_result, bytes_matrix);
    int *d_green_result;
    cudaMalloc(&d_green_result, bytes_matrix);
    int *d_red_result;
    cudaMalloc(&d_red_result, bytes_matrix);

    int num_threads = 16;
    dim3 threads(num_threads, num_threads);
    dim3 blocks(width / num_threads + 1, height / num_threads + 1);

    long start_time = clock();
    convolve <<< blocks, threads >>> (d_blue, d_blue_result, width, d_kernel, kernel_length);
    convolve <<< blocks, threads >>> (d_green, d_green_result, width, d_kernel, kernel_length);
    convolve <<< blocks, threads >>> (d_red, d_red_result, width, d_kernel, kernel_length);
    long end_time = clock();

    int *h_blue_result = (int*) malloc(bytes_matrix);
    cudaMemcpy(h_blue_result, d_blue_result, bytes_matrix, cudaMemcpyDeviceToHost);
    int *h_green_result = (int*) malloc(bytes_matrix);
    cudaMemcpy(h_green_result, d_green_result, bytes_matrix, cudaMemcpyDeviceToHost);
    int *h_red_result = (int*) malloc(bytes_matrix);
    cudaMemcpy(h_red_result, d_red_result, bytes_matrix, cudaMemcpyDeviceToHost);

    matrix_to_image(image, h_blue_result, h_green_result, h_red_result);
    cv::imwrite("output_cuda.jpg", image);

    double total_time = (double) (end_time - start_time) / CLOCKS_PER_SEC;
    printf("Convolution Time: %fs\n", total_time);

    free(h_kernel);
    free(h_blue);
    free(h_green);
    free(h_red);
    free(h_blue_result);
    free(h_green_result);
    free(h_red_result);
    cudaFree(d_kernel);
    cudaFree(d_blue);
    cudaFree(d_green);
    cudaFree(d_red);
    cudaFree(d_blue_result);
    cudaFree(d_green_result);
    cudaFree(d_red_result);

    return 0;
}

std::string get_image_name(std::string arg) {
    if(!arg.compare("harold.jpg") || !arg.compare("harold")) {
        return "../images/harold.jpg";
    }
    if(!arg.compare("christmas.jpg") || !arg.compare("xmas")) {
        return "../images/christmas.jpg";
    }
    if(!arg.compare("nier.jpg") || !arg.compare("nier")) {
        return "../images/nier.jpg";
    }
    if(!arg.compare("misha_mansoor.jpg") || !arg.compare("misha")) {
        return "../images/misha_mansoor.jpg";
    }
    if(!arg.compare("band.jpg") || !arg.compare("band")) {
        return "../images/band.jpg";
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

// fills the kernel with a circle, [k,k] center point
// assumes kernel is already malloc'd
// returns 1 if true, 0 if not
int inside_circle(int x, int y, int h, int k, int r) {
    if(pow(x + k, 2) + pow(y + k, 2) <= pow(r, 2)) {
        return 1;
    }
    return 0;
}

// inits kernel with a circle of 1s
void init_kernel(double *kernel, int kernel_length) {
    int k = kernel_length / 2;
    int sum = 0;

    for(int i = -k ; i <= k; i++) {
        for(int j = -k; j <= k; j++) {
            if(inside_circle(i, j, 0, 0, k)) {
                kernel[(i + k) * kernel_length + (j + k)] = 1;
                sum++;
            } else {
                kernel[(i + k) * kernel_length + (j + k)] = 0;
            }
        }
    }

    for(int i = 0; i < kernel_length; i++) {
        for(int j = 0; j < kernel_length; j++) {
            kernel[i * kernel_length + j] /= sum;
        }
    }
}

