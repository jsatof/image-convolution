#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <opencv2/opencv.hpp>

std::string get_image_name(std::string arg);
void init_kernel(double *kernel, int kernel_length);
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

            output[i * width + j] = sum / pow(kernel_length, 2);
        }
    }
}

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
    
    int *blue = (int*) malloc(height * width * sizeof(int));
    int *green = (int*) malloc(height * width * sizeof(int));
    int *red = (int*) malloc(height * width * sizeof(int));
    image_to_matrix(image, blue, green, red);

    int kernel_length = 7;
    double *kernel = (double*) malloc(pow(kernel_length, 2) * sizeof(double));
    init_kernel(kernel, kernel_length);

    int *blue_result = (int*) malloc(height * width * sizeof(int));
    int *green_result = (int*) malloc(height * width * sizeof(int));
    int *red_result = (int*) malloc(height * width * sizeof(int));

    // record start time
    long start_time = clock();
    convolve(blue, blue_result, height, width, kernel, kernel_length);
    convolve(green, green_result, height, width, kernel, kernel_length);
    convolve(red, red_result, height, width, kernel, kernel_length);
    long end_time = clock();

    matrix_to_image(image, blue_result, green_result, red_result);
    cv::imwrite("output_serial.jpg", image);

    double total_time = (double) (end_time - start_time) / CLOCKS_PER_SEC;
    printf("Convolution Time: %fs\n", total_time);

    free(blue);
    free(green);
    free(red);
    free(blue_result);
    free(green_result);
    free(red_result);
    free(kernel);
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

void init_kernel(double *kernel, int kernel_length) {
    for(int i = 0; i < kernel_length; i++) {
        for(int j = 0; j < kernel_length; j++) {
            kernel[i * kernel_length + j] = 1;
        }
    }
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