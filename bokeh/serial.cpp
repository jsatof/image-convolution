#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <opencv2/opencv.hpp>

void image_to_matrix(cv::Mat image, int *matrix);
void matrix_to_image(cv::Mat image, int *matrix);

int main() {
    std::string filename = "../images/harold.jpg";

    cv::Mat image = cv::imread(filename);
    int height = image.rows;
    int width = image.cols;

    int *image_matrix = (int*) malloc(height * width * sizeof(int));

    

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

// fills the horizontal components of the complex kernel
// assumes kernel is already malloc'd
void init_complex_kernel_x(int *kernel, int radius, int scale, double real_component, double imag_component) {
    int kernel_length = 2 * radius + 1;
    kernel
}