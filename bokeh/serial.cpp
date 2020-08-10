#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <complex.h>
#include <opencv2/opencv.hpp>

std::string get_image_name(std::string arg);
void image_to_matrix(cv::Mat image, int *matrix);
void matrix_to_image(cv::Mat image, int *matrix);
void init_kernel(int *kernel, int kernel_length);
void convolve(int *matrix, int height, int width, int *kernel, int kernel_length);

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

    cv::Mat image = cv::imread(filename);
    int height = image.rows;
    int width = image.cols;

    int *image_matrix = (int*) malloc(height * width * sizeof(int));
    image_to_matrix(image, image_matrix);
    
    int kernel_length = 25;
    int *kernel = (int*) malloc(kernel_length * kernel_length * sizeof(int));

    init_kernel(kernel, kernel_length);

    // 2D separated convolution
    int *h_buffer = (int*) malloc(width * sizeof(int));
    int *v_buffer = (int*) malloc(height * sizeof(int));

    

    



    matrix_to_image(image, image_matrix);
    cv::imwrite("harold.jpg", image);

    free(kernel);
    free(image_matrix);
    return 0;
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

bool inside_circle(int x, int y, int center_x, int center_y, int radius) {
    if(pow(x - center_x, 2) + pow(y - center_y, 2) <= pow(radius, 2)) {
        return true;
    }
    return false;
}

// fills the kernel with a circle
// assumes kernel is already malloc'd
void init_kernel(int *kernel, int kernel_length) {
    int radius = 5;
    int center_x = kernel_length / 2;
    int center_y = kernel_length / 2;

    for(int i = 0; i < kernel_length; i++) {
        for(int j = 0; j < kernel_length; j++) {
            if(inside_circle(i, j, center_x, center_y, radius)) {
                kernel[i * kernel_length + j] = 255;
            } else {
                kernel[i * kernel_length + j] = 0;
            }
        }
    }
}

int sum_kernel(int *kernel, int kernel_length) {
    int sum = 0;

    for(int i = 0; i < kernel_length; i++) {
        for(int j = 0; j < kernel_length; j++) {
            sum += kernel[i * kernel_length + j];
        }
    }

    return sum;
}

void convolve(int *matrix, int height, int width, int *kernel, int kernel_length) {
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {

            int sum = 0;
            int k = (kernel_length - 1) / 2;

            for(int u = -k; u <= k; u++) {
                for(int v = -k; v <= k; u++) {
                    if(i + u >= 0 && i + u < height && j + v >= 0 && j + v < width) {
                        sum += matrix[(i + u) * width + (j + v)] * kernel[(u + k) * kernel_length + (v + k)];
                    }
                }
            }

            int value = sum / pow(kernel_length, 2);
            matrix[i * width + j] = value;
        }
    }
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