#include <iostream>
#include <opencv2/opencv.hpp>
#include "image.hpp"

int main() {
    cv::Mat image = cv::imread("images/harold.jpg");

    std::cout << "Height: " << image.rows << "\tWidth: " << image.cols << std::endl;

    cuda_invert_color(image.data, image.rows, image.cols, image.channels());

    cv::imwrite("images/new_harold.jpg", image);

    return 0;
}

