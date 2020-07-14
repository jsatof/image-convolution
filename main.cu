#include <iostream>
#include <vector>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "image.hpp"

#define CV_LOAD_IMAGE_COLOR 1

__global__ void apply_gaussian(unsigned char *image, int *kernel, int kernel_length, int channels) {
    
}

// returns value for kernel at index x,y with omega of k, gaussian formula
double gaussian(double x, double y, double sigma) {
    return (1.0 / (2.0 * M_PI * pow(sigma, 2.0))) * exp(-(pow(x, 2.0) + pow(y, 2.0)) / (2 * pow(sigma, 2.0)));
}

// initializes kernel with gaussian values, uses k as sigma
void init_kernel(double *kernel, int kernel_length, int k) {
    for(int x = -k; x <= k; x++) {
        for(int y = -k; y <= k; y++) {
            kernel[(x + k) * kernel_length + (y + k)] = gaussian(x, y, 10);
        }
    }
}

int main() {
    std::string periphery = "images/periphery.jpg";

    cv::Mat image = cv::imread(periphery, CV_LOAD_IMAGE_COLOR);

    int K = 10;
    int kernel_length = 2 * K + 1;

    double *kernel = (double*) malloc(pow(kernel_length, 2) * sizeof(double));
    init_kernel(kernel, kernel_length, K);

    try {
        CV_Assert(image.channels() == 4);
        for(int i = 0; i < image.rows; i++) {
            for(int j = 0; j < image.cols; j++) {
                cv::Vec4b& bgra = image.at<cv::Vec4b>(i, j);
                bgra[3] = 100;          // alpha
            }
        }
    } catch(cv::Exception& e) {
        std::cout << e.what() << std::endl;
    }


    cv::imwrite("images/new_periphery.jpg", image);

    return 0;
}
