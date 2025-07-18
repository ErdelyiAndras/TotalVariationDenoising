#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include "../Image/Image.h"
#include "Denoising.h"

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0]
            << " <input_image_path> <output_image_path> <strength> <step_size> <tol> <suppress_log>"
            << std::endl;
        return -1;
    }
    std::string suppress_log_str = argv[6];
    bool suppress_log = true;
    if (suppress_log_str == "false" || suppress_log_str == "0") {
        suppress_log = false;
    }

    try {
        Image image(argv[1]);

        float strength = std::stof(argv[3]);
        float step_size = std::stof(argv[4]);
        float tol = std::stof(argv[5]);

        auto start = std::chrono::high_resolution_clock::now();

        Image denoisedImage = tv_denoise_gradient_descent(image, strength, step_size, tol, suppress_log);

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<float> elapsed = end - start;
        std::cout << "CPU_Denoising took: " << elapsed.count() << " seconds" << std::endl;

        cv::Mat displayImage = denoisedImage.toMat();
        
        if (!suppress_log) {
            cv::imshow("Denoised", displayImage);
            cv::waitKey(0);
        }

        std::string path = argv[2];
        cv::imwrite(path, displayImage);
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }
    catch (...) {
        std::cout << "Other exception" << std::endl;
        return -1;
    }

    return 0;
}
