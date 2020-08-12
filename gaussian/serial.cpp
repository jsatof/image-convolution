#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <opencv2/opencv.hpp>

#define PI acos(-1)

std::string get_image_name(std::string arg);
void image_to_matrix(cv::Mat image, int *blue_matrix, int *green_matrix, int *red_matrix);
void matrix_to_image(cv::Mat image, int *blue_matrix, int *green_matrix, int *red_matrix);
void init_kernel(double *kernel, int kernel_length);
void normalize_kernel(double *kernel, int kernel_length);
double gaussian_value(int x, int y, int sigma);

void convolve(int *input, int *output, int height, int width, double *kernel, int kernel_length) {
    int k = kernel_length / 2;

    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            
            double sum = 0;

            for(int u = -k; u <= k; u++) {
                for(int v = -k; v <= k; v++) {
                    if(i + u < height && j + v < width && i + u >= 0 && j + v >= 0) {
                        double kernel_value = kernel[(u + k) * kernel_length + (v + k)];
                        int matrix_value = input[(i + u) * width + (j + v)];

                        sum += matrix_value * kernel_value;
                    }
                }
            }

            output[i * width + j] = (int) sum;
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

    int *blue_matrix = (int*) malloc(height * width * sizeof(int));
    int *green_matrix = (int*) malloc(height * width * sizeof(int));
    int *red_matrix = (int*) malloc(height * width * sizeof(int));
    image_to_matrix(image, blue_matrix, green_matrix, red_matrix);

    int kernel_length = 7;
    double *kernel = (double*) malloc(pow(kernel_length, 2) * sizeof(double));
    init_kernel(kernel, kernel_length);
    normalize_kernel(kernel, kernel_length);

    int *blue_result = (int*) malloc(height * width * sizeof(int));
    int *green_result = (int*) malloc(height * width * sizeof(int));
    int *red_result = (int*) malloc(height * width * sizeof(int));

    // record convolve time
    long start_time = clock();
    convolve(blue_matrix, blue_result, height, width, kernel, kernel_length);
    convolve(green_matrix, green_result, height, width, kernel, kernel_length);
    convolve(red_matrix, red_result, height, width, kernel, kernel_length);
    long end_time = clock();

    double total_time = (double) (end_time - start_time) / CLOCKS_PER_SEC;
    printf("Convolution Time: %fs\n", total_time);

    matrix_to_image(image, blue_result, green_result, red_result);
    cv::imwrite("output_serial.jpg", image);

    free(kernel);
    free(blue_matrix);
    free(green_matrix);
    free(red_matrix);
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

double gaussian_value(int x, int y, int sigma) {
    return exp(-(pow(x,2) + pow(y,2)) / (2 * pow(sigma,2))) / (2 * PI * pow(sigma, 2));
}

void init_kernel(double *kernel, int kernel_length) {
    int sigma = 10;
    int k = kernel_length / 2;

    for(int i = -k; i <= k; i++) {
        for(int j = -k; j <= k ; j++) {
            kernel[(i + k) * kernel_length + (j + k)] = gaussian_value(i, j, sigma);
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

void image_to_matrix(cv::Mat image, int *blue_matrix, int *green_matrix, int *red_matrix) {
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            blue_matrix[i * image.cols + j] = image.at<cv::Vec3b>(i, j)[0];
            green_matrix[i * image.cols + j] = image.at<cv::Vec3b>(i, j)[1];
            red_matrix[i * image.cols + j] = image.at<cv::Vec3b>(i, j)[2];
        }
    }
}

void matrix_to_image(cv::Mat image, int *blue_matrix, int *green_matrix, int *red_matrix) {
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            image.at<cv::Vec3b>(i, j)[0] = blue_matrix[i * image.cols + j];
            image.at<cv::Vec3b>(i, j)[1] = green_matrix[i * image.cols + j];
            image.at<cv::Vec3b>(i, j)[2] = red_matrix[i * image.cols + j];
        }
    }
}