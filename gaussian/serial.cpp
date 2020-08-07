#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <opencv2/opencv.hpp>

#define PI acos(-1)

void init_colors(cv::Mat image, int *blue_matrix, int *green_matrix, int *red_matrix, int height, int width);
void set_image_colors(cv::Mat image, int *blue_matrix, int *green_matrix, int *red_matrix, int height, int width);
float gaussian_value(int x, int y, int sigma);
void init_kernel_2d(float *kernel, int kernel_length);
void init_kernel_h(float *kernel, int kernel_length);
void init_kernel_v(float *kernel, int kernel_length);
void convolve(int *blue_matrix, int *green_matrix, int *red_matrix, int height, int width, float *kernel, int kernel_length);

int main() {
    cv::Mat image = cv::imread("../images/christmas.jpg");
    int height = image.rows;
    int width = image.cols;
    printf("Image Dims: %dx%d\n", width, height);

    int kernel_length = 3;
    float *kernel = (float*) malloc(pow(kernel_length, 2) * sizeof(float));
    
    printf("Initializing Kernel\n");
    init_kernel_2d(kernel, kernel_length);

    int *blue_matrix = (int*) malloc(height * width * sizeof(int));
    int *green_matrix = (int*) malloc(height * width * sizeof(int));
    int *red_matrix = (int*) malloc(height * width * sizeof(int));

    printf("Initializing Matrices\n");
    init_colors(image, blue_matrix, green_matrix, red_matrix, height, width);

    printf("Convolving Image\n");
    long start_time = clock();
    convolve(blue_matrix, green_matrix, red_matrix, height, width, kernel, kernel_length);
    long end_time = clock();

    double total_time = (double) (end_time - start_time) / CLOCKS_PER_SEC;
    printf("Convolution Time: %fs\n");


/*
    for(int i = 0; i < kernel_length; i++) {
        for(int j = 0; j < kernel_length; j++) {
            printf("%f ", kernel[i * kernel_length + j]);
        }
        printf("\n");
    }*/

    printf("Writing New Image\n");
    set_image_colors(image, blue_matrix, green_matrix, red_matrix, height, width);

    cv::imwrite("christmas.jpg", image);

    free(blue_matrix);
    free(green_matrix);
    free(red_matrix);
    free(kernel);
}

void init_colors(cv::Mat image, int *blue_matrix, int *green_matrix, int *red_matrix, int height, int width) {
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            blue_matrix[i * width + j] = image.at<cv::Vec3b>(i, j)[0];
            green_matrix[i * width + j] = image.at<cv::Vec3b>(i, j)[1];
            red_matrix[i * width + j] = image.at<cv::Vec3b>(i, j)[2];
        }
    }
}

void set_image_colors(cv::Mat image, int *blue_matrix, int *green_matrix, int *red_matrix, int height, int width) {
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            image.at<cv::Vec3b>(i, j)[0] = blue_matrix[i * width + j];
            image.at<cv::Vec3b>(i, j)[1] = green_matrix[i * width + j];
            image.at<cv::Vec3b>(i, j)[2] = red_matrix[i * width + j];
        }
    }
}


float gaussian_value(int x, int y, int sigma) {
    return exp(-(pow(x,2) + pow(y,2)) / (2 * pow(sigma,2))) / (2 * PI * pow(sigma, 2));
}

void init_kernel_2d(float *kernel, int kernel_length) {
    int sigma = 1;
    int k = (kernel_length - 1) / 2;

    for(int i = -k; i <= k; i++) {
        for(int j = -k; j <= k ; j++) {
            kernel[(i + k) * kernel_length + (j + k)] = gaussian_value(i, j, sigma);
        }
    }
}

void init_kernel_h(float *kernel, int kernel_length) {

}

void init_kernel_v(float *kernel, int kernel_length) {

}

void convolve(int *blue_matrix, int *green_matrix, int *red_matrix, int height, int width, float *kernel, int kernel_length) {
    int k = (kernel_length - 1) / 2;
    int kernel_size = pow(kernel_length, 2);
    
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {

            int blue_sum = 0;
            int green_sum = 0;
            int red_sum = 0;

            for(int u = -k; u <= k; u++) {
                for(int v = -k; v <= k; v++) {
                    if(i + u >= 0 && j + v >= 0 && i + u < width && j + v < height) {
                        blue_sum += blue_matrix[(i + u) * width + (j + v)] * kernel[(u + k) * kernel_length + (v + k)];
                        green_sum += green_matrix[(i + u) * width + (j + v)] * kernel[(u + k) * kernel_length + (v + k)];
                        red_sum += red_matrix[(i + u) * width + (j + v)] * kernel[(u + k) * kernel_length + (j + v)];
                    }
                }
            }

            blue_matrix[i * width + j] = blue_sum;
            green_matrix[i * width + j] = green_sum;
            red_matrix[i * width + j] = red_sum;
        }
    }
}