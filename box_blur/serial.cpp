#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <opencv2/opencv.hpp>

void check_pixel(int *value) {
    if(*value > 255)
        *value = 255;
    if(*value < 0)
        *value = 0;
}

int main(int argc, char **argv) {
    cv::Mat image = cv::imread("../images/harold.jpg");

    int kernel_length = 11;
    unsigned char *kernel = (unsigned char*) malloc(pow(kernel_length, 2) * sizeof(unsigned char));

    for(int i = 0; i < pow(kernel_length, 2); i++) {
        kernel[i] = 1;
    }

    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            
            int blue_sum = 0;
            int green_sum = 0;
            int red_sum = 0;
            for(int u = -5; u <= 5; u++) {
                for(int v = -5; v <= 5; v++) {
                    if(i + u < image.rows && j + v < image.cols &&
                       i + u >= 0 && j + v >= 0) {

                        blue_sum  += kernel[(u + 5) * kernel_length + (v + 5)] * image.at<cv::Vec3b>(i + u, j + v)[0];
                        green_sum += kernel[(u + 5) * kernel_length + (v + 5)] * image.at<cv::Vec3b>(i + u, j + v)[1];
                        red_sum   += kernel[(u + 5) * kernel_length + (v + 5)] * image.at<cv::Vec3b>(i + u, j + v)[2]; 
                    }
                }
            }

            // divide by length of kernel
            int blue_value  = blue_sum  / pow(kernel_length, 2);
            int green_value = green_sum / pow(kernel_length, 2);
            int red_value   = red_sum   / pow(kernel_length, 2);

            check_pixel(&blue_value);
            check_pixel(&green_value);
            check_pixel(&red_value);

            image.at<cv::Vec3b>(i, j) = cv::Vec3b(blue_value, green_value, red_value);
        }
    }
    
    cv::imwrite("harold.jpg", image);

    free(kernel);
    return 0;
}
