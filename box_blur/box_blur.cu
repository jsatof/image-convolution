#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

// fast memory block on gpu, 5x5 kernel
__constant__ double device_kernel[25];

// prototype for helper functions
std::string get_image_name(std::string arg);
__global__ void convolve(int *matrix, int *result_matrix, int height, int width, double *device_kernel, int kernel_length);
void init_kernel(double *kernel, int kernel_length);
void normalize_kernel(double *kernel, int kernel_length);
void get_image_matrix(cv::Mat image, int *matrix); 
void set_image_matrix(cv::Mat image, int *matrix);


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
    cv::Mat grey_image;
    cv::cvtColor(color_image, grey_image, cv::COLOR_BGR2GRAY);

    int height = grey_image.rows;
    int width = grey_image.cols;
    printf("Image Dims: %dx%d\n", width, height);


    // CPU initializing stuff on Host
    printf("Initializing Kernel\n");
    int k = 2;
    int kernel_length = 2 * k + 1;
    size_t bytes_kernel = pow(kernel_length, 2) * sizeof(double);
    double *kernel = (double*) malloc(bytes_kernel);

    init_kernel(kernel, kernel_length); 
    normalize_kernel(kernel, kernel_length); 

    size_t bytes_matrix = height * width * sizeof(int);
    int *image_matrix = (int*) malloc(bytes_matrix);
    set_image_matrix(grey_image, image_matrix); 


    // copy kernel to __constant__ gpu block
    cudaMemcpyToSymbol(device_kernel, kernel, bytes_kernel);
    
    // copy matrix to gpu
    int *device_image_matrix;
    cudaMalloc(&device_image_matrix, bytes_matrix);
    cudaMemcpy(device_image_matrix, image_matrix, bytes_matrix, cudaMemcpyHostToDevice);

    int *device_result_matrix;
    cudaMalloc(&device_result_matrix, bytes_matrix);

    // set cuda grid
    int num_threads = 16;
    int num_blocks_x = (width + num_threads - 1) / num_threads;  
    int num_blocks_y = (height + num_threads - 1) / num_threads; 

    dim3 block_dim(num_threads, num_threads);   // each block has num_threads threads in x and y dimensions
    dim3 grid_dim(num_blocks_x, num_blocks_y);  // a grid will have num_blocks blocks in x and y dimensions
     
    // perform convolution
    long start_time = clock();
    convolve<<< grid_dim, block_dim >>>(device_image_matrix, device_result_matrix, height, width, device_kernel, kernel_length);
    long end_time = clock();

    double total_time = (double) (end_time - start_time) / CLOCKS_PER_SEC;
    printf("Convolution Time: %fs\n", total_time);

    // write to new image
    int *result_matrix = (int*) malloc(bytes_matrix);
    cudaMemcpy(result_matrix, device_result_matrix, bytes_matrix, cudaMemcpyDeviceToHost);

    set_image_matrix(grey_image, result_matrix); 
    cv::imwrite("output_cuda.jpg", grey_image);
    
    cudaFree(device_image_matrix);
    cudaFree(device_kernel);
    cudaFree(device_result_matrix);
    free(kernel);
    free(image_matrix);
    free(result_matrix);
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

__global__ void convolve(int *matrix, int *result_matrix, int height, int width, double *kernel, int kernel_length) {
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int y_index = blockIdx.y * blockDim.y + threadIdx.y;

    int k = kernel_length / 2;
    double sum = 0;

    for(int i = -k; i <= k; i++) {
        for(int j = -k; j <= k; j++) {

            if(i + x_index >= 0 && i + x_index < width) {
                if(j + y_index >= 0 && j + y_index < height) {
                    sum += matrix[(i + x_index) * width + (j + y_index)] * kernel[(i + k) * kernel_length + (j + k)];
                }
            }

        }
    }

    double pixel_value = sum;
    if(pixel_value > 255) {
        pixel_value = 255;
    }
    if(pixel_value < 0) {
        pixel_value = 0;
    }

    result_matrix[x_index * width + y_index] = (int) pixel_value;
}

void init_kernel(double *kernel, int kernel_length) {
    for(int i = 0; i < kernel_length; i++) {
        for(int j = 0; j < kernel_length; j++) {
            kernel[i * kernel_length + j] = 1;
        }
    }
}

void normalize_kernel(double *kernel, int kernel_length) {
    double sum = 0;
    for(int i = 0; i < kernel_length; i++) {
        for(int j = 0; j < kernel_length; j++) {
            sum += kernel[i * kernel_length + j];
        }
    }

    for(int i = 0; i < kernel_length; i++) {
        for(int j = 0; j < kernel_length; j++) {
            kernel[i * kernel_length + j] /= sum;
        }
    }
}

void get_image_matrix(cv::Mat image, int *matrix) {
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            matrix[i * image.cols + j] = image.at<uchar>(i, j);
        }
    }
}

void set_image_matrix(cv::Mat image, int *matrix) {
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            image.at<uchar>(i, j) = matrix[i * image.cols + j];
        }
    }
}