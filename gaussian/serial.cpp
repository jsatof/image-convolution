#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <opencv2/opencv.hpp>

#define PI acos(-1)

std::string get_image_name(std::string arg);
void check_pixel(double value);
void get_image_colors(cv::Mat image, int *blue_matrix, int *green_matrix, int *red_matrix);
void set_image_colors(cv::Mat image, int *blue_matrix, int *green_matrix, int *red_matrix);
void init_kernel(double *kernel, int kernel_length);
void normalize_kernel(double *kernel, int kernel_length);
double gaussian_value(int x, int y, int sigma);
void convolve(int *blue_matrix, int *green_matrix, int *red_matrix, int height, int width, double *kernel, int kernel_length);


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


    int k = 6;
    int kernel_length = 2 * k + 1;
    double *kernel = (double*) malloc(pow(kernel_length, 2) * sizeof(double));

    init_kernel(kernel, kernel_length);
    normalize_kernel(kernel, kernel_length);

    int *blue_matrix = (int*) malloc(height * width * sizeof(int));
    int *green_matrix = (int*) malloc(height * width * sizeof(int));
    int *red_matrix = (int*) malloc(height * width * sizeof(int));

    get_image_colors(image, blue_matrix, green_matrix, red_matrix);

    // record convolve time
    long start_time = clock();
    convolve(blue_matrix, green_matrix, red_matrix, height, width, kernel, kernel_length);
    long end_time = clock();

    double total_time = (double) (end_time - start_time) / CLOCKS_PER_SEC;
    std::cout << "Convolution Time: " << total_time << std::endl;


    set_image_colors(image, blue_matrix, green_matrix, red_matrix);
    cv::imwrite("output_serial.jpg", image);

    free(kernel);
    free(blue_matrix);
    free(green_matrix);
    free(red_matrix);
    return 0;
}

void convolve(int *blue_matrix, int *green_matrix, int *red_matrix, int height, int width, double *kernel, int kernel_length) {
    int k = kernel_length / 2;

    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            
            double blue_sum = 0;
            double green_sum = 0;
            double red_sum = 0;
            for(int u = -k; u <= k; u++) {
                for(int v = -k; v <= k; v++) {
                    if(i + u < height && j + v < width &&
                       i + u >= 0 && j + v >= 0) {

                        blue_sum  += kernel[(u + 5) * kernel_length + (v + 5)] *  blue_matrix[(i + u) * width + (j + v)];
                        green_sum += kernel[(u + 5) * kernel_length + (v + 5)] * green_matrix[(i + u) * width + (j + v)];
                        red_sum   += kernel[(u + 5) * kernel_length + (v + 5)] *   red_matrix[(i + u) * width + (j + v)]; 
                    }
                }
            }

            double blue_value  = blue_sum;
            double green_value = green_sum;
            double red_value   = red_sum;

            check_pixel(blue_value);
            check_pixel(green_value);
            check_pixel(red_value);

            blue_matrix[i * width + j] = blue_value;
            green_matrix[i * width + j] = green_value;
            red_matrix[i * width + j] = red_value;
        }
    }
}

void check_pixel(double value) {
    if(value > 255)
        value = 255;
    if(value < 0)
        value = 0;
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
    int sigma = 5;
    int k = (kernel_length - 1) / 2;

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