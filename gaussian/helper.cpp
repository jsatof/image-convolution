#include "helper.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>

#define PI acos(-1)

void get_image_colors(cv::Mat image, int *blue_matrix, int *green_matrix, int *red_matrix) {
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            blue_matrix[i * image.cols + j] = image.at<cv::Vec3b>(i, j)[0];
            green_matrix[i * image.cols + j] = image.at<cv::Vec3b>(i, j)[1];
            red_matrix[i * image.cols + j] = image.at<cv::Vec3b>(i, j)[2];
        }
    }
}

void set_image_colors(cv::Mat image, int *blue_matrix, int *green_matrix, int *red_matrix) {
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            image.at<cv::Vec3b>(i, j)[0] = blue_matrix[i * image.cols + j];
            image.at<cv::Vec3b>(i, j)[1] = green_matrix[i * image.cols + j];
            image.at<cv::Vec3b>(i, j)[2] = red_matrix[i * image.cols + j];
        }
    }
}

float gaussian_value(int x, int y, int sigma) {
    return exp(-(pow(x,2) + pow(y,2)) / (2 * pow(sigma,2))) / (2 * PI * pow(sigma, 2));
}

// fills with values and normalizes
void init_kernel(float *kernel, int kernel_length) {
    int sigma = 1;
    int k = (kernel_length - 1) / 2;
    float sum = 0;

    for(int i = -k; i <= k; i++) {
        for(int j = -k; j <= k ; j++) {
            kernel[(i + k) * kernel_length + (j + k)] = gaussian_value(i, j, sigma);
        }
    }
}

void normalize_kernel(float *kernel, int kernel_length) {
    float sum = 0;

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

void check_pixel_value(int value) {
    if(value < 0) 
        value = 0;
    if(value > 255)
        value = 255;
}

void convolve(int *blue_matrix, int *green_matrix, int *red_matrix, int height, int width, float *kernel, int kernel_length) {
    int k = (kernel_length - 1) / 2;
    int kernel_size = pow(kernel_length, 2);
    
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {

            float blue_sum = 0;
            float green_sum = 0;
            float red_sum = 0;

            for(int u = -k; u <= k; u++) {
                for(int v = -k; v <= k; v++) {
                    if(i + u >= 0 && j + v >= 0 && i + u < width && j + v < height) {

                        blue_sum  +=  blue_matrix[(i + u) * width + (j + v)] * kernel[(u + k) * kernel_length + (v + k)];
                        green_sum += green_matrix[(i + u) * width + (j + v)] * kernel[(u + k) * kernel_length + (v + k)];
                        red_sum   +=   red_matrix[(i + u) * width + (j + v)] * kernel[(u + k) * kernel_length + (j + v)];

                    }
                }
            }

            int blue_value  =  blue_sum;
            int green_value = green_sum;
            int red_value   =   red_sum;

            check_pixel_value(blue_value);
            check_pixel_value(green_value);
            check_pixel_value(red_value);

            blue_matrix [i * width + j] = blue_value;
            green_matrix[i * width + j] = green_value;
            red_matrix  [i * width + j] = red_value;
        }
    }
}