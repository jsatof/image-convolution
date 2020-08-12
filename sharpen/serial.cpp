#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <opencv2/opencv.hpp>

std::string get_image_name(std::string arg);
void image_to_matrix(cv::Mat image, int *blue, int *green, int *red);
void matrix_to_image(cv::Mat image, int *blue, int *green, int *red);

void convolve(int *input, int *output, int height, int width, double *kernel, int kernel_length) {
    int k = kernel_length / 2;
    
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {

            double sum = 0.0;

            for(int u = -k; u <= k; u++) {
                for(int v = -k; v <= k; v++) {
                    if(i + u >= 0 && i + u < height && j + v >= 0 && j + v < width) {
                        double kernel_value = kernel[(u + k) * kernel_length + (v + k)];
                        int matrix_value = input[(i + u) * width + (j + v)];

                        sum += matrix_value * kernel_value;
                    }
                } 
            }

            output[i * width + j] = sum;

        }
    }
}

int main(int argc, char **argv) {
    if(argc != 2) {
        printf("usage: ./sharpen <name of image>\n");
        return 1;
    }

    std::string filename = get_image_name(argv[1]);

    if(filename.compare("invalid") == 0) {
        printf("Invalid image\n");
        return 1;
    }

    cv::Mat image = cv::imread(filename);
    int height = image.rows;
    int width = image.cols;

    size_t bytes_image = height * width * sizeof(int);
    int *blue = (int*) malloc(bytes_image);
    int *green = (int*) malloc(bytes_image);
    int *red = (int*) malloc(bytes_image);
    image_to_matrix(image, blue, green, red);

    int kernel_length = 3;
    double kernel[9] = {
        0.0, -1.0, 0.0,
        -1.0, 5.0, -1.0,
        0.0, -1.0, 0.0
    };

    int *blue_result = (int*) malloc(bytes_image);
    int *green_result = (int*) malloc(bytes_image);
    int *red_result = (int*) malloc(bytes_image);

    long start_time = clock();
    convolve(blue, blue_result, height, width, kernel, kernel_length);
    convolve(green, green_result, height, width, kernel, kernel_length);
    convolve(red, red_result, height, width, kernel, kernel_length);
    long end_time = clock();

    matrix_to_image(image, blue_result, green_result, red_result);
    cv::imwrite("output_serial.jpg", image);

    double conv_time = (double) (end_time - start_time) / CLOCKS_PER_SEC;
    printf("Convolution Time: %fs\n", conv_time);

    free(blue);
    free(green);
    free(red);
    free(blue_result);
    free(green_result);
    free(red_result);

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
    if(arg.compare("nier.jpg") == 0 || arg.compare("nier") == 0) {
        return "../images/nier.jpg";
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