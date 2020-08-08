#ifndef _HELPER_FUNC
#define _HELPER_FUNC
#include <opencv2/opencv.hpp>

void get_image_colors(cv::Mat image, int *blue_matrix, int *green_matrix, int *red_matrix);
void set_image_colors(cv::Mat image, int *blue_matrix, int *green_matrix, int *red_matrix);

void init_kernel(float *kernel, int kernel_length);
void normalize_kernel(float *kernel, int kernel_length);

float gaussian_value(int x, int y, int sigma);

void convolve(int *blue_matrix, int *green_matrix, int *red_matrix, int height, int width, float *kernel, int kernel_length);

#endif