#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include "../Image/Image.h"
#include "Denoising.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    Image image(argv[1]);

    auto start = std::chrono::high_resolution_clock::now();

    Image denoisedImage = tv_denoise_gradient_descent(image, 0.1f);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float> elapsed = end - start;
    std::cout << "CPU_Denoising took: " << elapsed.count() << " seconds" << std::endl;

    cv::Mat displayImage = denoisedImage.toMat();

    cv::imshow("Denoised", displayImage);
    cv::waitKey(0);

    std::string path = argv[1];
    size_t last_dot = path.find_last_of('.');
    size_t last_slash = path.find_last_of("/\\");
    if (last_dot == std::string::npos || (last_slash != std::string::npos && last_dot < last_slash)) {
        path = path + "-denoised-cpu";
    }
    path = path.substr(0, last_dot) + "-denoised-cpu" + path.substr(last_dot);

    cv::imwrite(path, displayImage);

    return 0;
}
