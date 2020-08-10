#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

std::string get_image_name(std::string arg);
void convolve(int *matrix, int height, int width, int *kernel, int k);
void reduce_matrix(int *matrix_a, int *matrix_b, int *result_matrix, int height, int width);
void image_to_matrix(cv::Mat image, int *matrix);
void matrix_to_image(cv::Mat image, int *matrix);

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

    int kernel_length = 3;

    int sobel_x[9] = {
        1, 0, -1,
        2, 0, -2,
        1, 0, -1
    };

    int sobel_y[9] = {
        1, 2, 1,
        0, 0, 0,
        -1, -2, -1
    };

    long start_time = clock();
    convolve(matrix_x, height, width, sobel_x, kernel_length);
    convolve(matrix_y, height, width, sobel_y, kernel_length);
    long end_time = clock();

    double total_time = (double) (end_time - start_time) / CLOCKS_PER_SEC;
    std::cout << "Convolution Time: " << total_time << "s" << std::endl;

    int *result_matrix = (int*) malloc(height * width * sizeof(int));
    reduce_matrix(matrix_x, matrix_y, result_matrix, height, width);

    matrix_to_image(grey_image, result_matrix);
    cv::imwrite("output_serial.jpg", grey_image);

    free(matrix_x);
    free(matrix_y);
    free(result_matrix);
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

void convolve(int *matrix, int height, int width, int *kernel, int kernel_length) {
    int k = kernel_length / 2;

    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {

            double sum = 0;

            for(int u = -k; u <= k; u++) {
                for(int v = -k; v <= k; v++) {

                    if(i + u >= 0 && i + u < height &&
                       j + v >= 0 && j + v < width) {

                           sum += kernel[(u + k) * kernel_length + (v + k)] * matrix[(i + u) * width + (j + v)];
                    }

                }
            }

            double pixel_value = sum / pow(kernel_length, 2);
            if(pixel_value < 0)
                pixel_value = 0;
            if(pixel_value > 255)
                pixel_value = 255;

            matrix[i * width + j] =  (int) pixel_value;

        }
    }
}

// assumes all matrices are equal size
// uses pythagorean theorem to get magnitudes of the 2 input matrices
void reduce_matrix(int *matrix_a, int *matrix_b, int *result_matrix, int height, int width) {
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            int a = matrix_a[i * width + j];
            int b = matrix_b[i * width + j];

            result_matrix[i * width + j] = (int) sqrtf(a * a + b * b);
        }
    }
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
