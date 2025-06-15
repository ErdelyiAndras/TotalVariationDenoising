// CPU_Denoising.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "Image.h"

int main(int argc, char** argv) {
    if (argc != 2) {
		std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

	Image image(argv[1]);

    std::cout << "Image loaded: " << image.getRows() << "x" << image.getCols() << std::endl;

    for (int i = 0; i < image.getRows(); ++i) {
        for (int j = 0; j < image.getCols(); ++j) {
            if (i < 200 && j > 200) {
                image(i, j) = 50;
            }
        }
    }

	cv::Mat displayImage = image.toMat();

    cv::imshow("Display", displayImage);
    cv::waitKey(0);
	
    std::string path = argv[1];
    size_t last_dot = path.find_last_of('.');
    size_t last_slash = path.find_last_of("/\\");
    if (last_dot == std::string::npos || (last_slash != std::string::npos && last_dot < last_slash)) {
        path = path + "-denoised";
    }
    path = path.substr(0, last_dot) + "-denoised" + path.substr(last_dot);

	cv::imwrite(path, displayImage);

    return 0;
}


// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
