#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

std::string get_image_name(std::string arg);
void image_to_matrix(cv::Mat image, int *matrix);
void matrix_to_image(cv::Mat image, int *matrix);

void convolve(int *input, int *output, int height, int width, double *kernel, int kernel_length) {
    int k = kernel_length / 2;

    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {

            double sum = 0;

            for(int u = -k; u <= k; u++) {
                for(int v = 0; v < kernel_length; v++) {
                    if(i + u >= 0 && i + u < height) {
                        if(j + v >= 0 && j + v < width) {
                            sum += input[(i + u) * width + (j + v)] * kernel[(u + k) * kernel_length + (v + k)];
                        }
                    }
                }
            }

            output[i * width + j] =  (int) sum;

        }
    }
}

// assumes all matrices are equal size
// uses pythagorean theorem to get magnitudes of the 2 input matrices
void calc_magnitude(int *matrix_a, int *matrix_b, int *result_matrix, int height, int width) {
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            int a = matrix_a[i * width + j];
            int b = matrix_b[i * width + j];

            result_matrix[i * width + j] = (int) sqrtf(a * a + b * b);
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

    cv::Mat grey_image; 
    cv::cvtColor(image, grey_image, cv::COLOR_BGR2GRAY);

    int *matrix_x = (int*) malloc(height * width * sizeof(int));
    int *matrix_y = (int*) malloc(height * width * sizeof(int));

    // fill matrix with image's values
    image_to_matrix(grey_image, matrix_x);
    image_to_matrix(grey_image, matrix_y);

    int kernel_length = 5;

    // these kernels are not my discovery. 
    // courtesy of https://stackoverflow.com/questions/9567882/sobel-filter-kernel-of-large-size/41065243#41065243
    double sobel_x[25] = {
        -2/8, -1/5,  0,  1/5,  2/8,
        -2/5, -1/2,  0,  1/2,  2/5,
        -2/4, -1/1,  0,  1/1,  2/4,
        -2/5, -1/2,  0,  1/2,  2/5,
        -2/8, -1/5,  0,  1/5,  2/8
    };

    double sobel_y[25] = {
        -2/8, -2/5, -2/4, -2/5, -2/8,
        -1/5, -1/2, -1/1, -1/2, -1/5,
        0,    0,    0,    0,    0,
        1/5, 1/2, 1/1, 1/2, 1/5,
        -2/8, -2/5, -2/4, -2/5, -2/8,
    };

    int *result_x = (int*) malloc(height * width * sizeof(int));
    int *result_y = (int*) malloc(height * width * sizeof(int));
    int *combined_result = (int*) malloc(height * width * sizeof(int));

    long start_time = clock();
    convolve(matrix_x, result_x, height, width, sobel_x, kernel_length);
    convolve(matrix_y, result_y, height, width, sobel_y, kernel_length);
    calc_magnitude(result_x, result_y, combined_result, height, width);
    long end_time = clock();

    double total_time = (double) (end_time - start_time) / CLOCKS_PER_SEC;
    printf("Convolution Time: %fs\n", total_time);

    matrix_to_image(grey_image, combined_result);
    cv::imwrite("output_serial.jpg", grey_image);

    free(matrix_x);
    free(matrix_y);
    free(result_x);
    free(result_y);
    free(combined_result);
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
