// CPU_Denoising.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "Image.h"
#include <algorithm>
#include <cmath>


Image totalVariationDenoise(const Image& img, long double lambda = 0.125, long double tau = 0.125, int iterations = 50) {
    int rows = img.getRows();
    int cols = img.getCols();
    

    Image u0 = img;
    Image u = img;
    Image returnImage = img;

	int allIterations = iterations * rows * cols;
    int zeroUpdateCount = 0;
    for (int iter = 0; iter < iterations; ++iter) {
        for (int i = 1; i < rows - 1; ++i) {
            for (int j = 1; j < cols - 1; ++j) {

                double ux = u(i, j + 1) - u(i, j);
                double uy = u(i + 1, j) - u(i, j);
                double norm = std::sqrt(ux * ux + uy * uy) + 1e-8;

                double div = (u(i, j) - u(i, j - 1)) + (u(i, j) - u(i - 1, j));

                long double rhs = tau * (u0(i, j) - u(i, j) + lambda * div / norm);
                if (rhs == 0) {

					//std::cout << "zero rhs at (" << i << ", " << j << ") in iteration " << iter << ", skipping update." << std::endl;
					zeroUpdateCount++;
                }
				//std::cout << "Iteration: " << iter << ", Pixel: (" << i << ", " << j << "), Norm: " << norm
					//<< ", Divergence: " << div << ", Update: " << rhs << std::endl;
                returnImage(i, j) = u(i, j) + tau * (u0(i, j) - u(i, j) + lambda * div / norm);
                returnImage(i, j) = std::max(0.0L, std::min(255.0L, returnImage(i, j)));
            }
        }
        u = returnImage;
    }

	std::cout << "Total iterations: " << allIterations << ", Zero updates: " << zeroUpdateCount << std::endl;
    return returnImage;
}

int main(int argc, char** argv) {
    if (argc != 2) {
		std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

	Image image(argv[1]);

    std::cout << "Image loaded: " << image.getRows() << "x" << image.getCols() << std::endl;

	std::cout << "Starting denoising." << std::endl;
	Image denoisedImage = totalVariationDenoise(image, 1.814, 2.023, 50);
	std::cout << "Denoising completed." << std::endl;

	cv::Mat displayImage = denoisedImage.toMat();
    
    cv::imshow("Denoised Image", displayImage);
    cv::waitKey(0);

    int samePixelCount = 0;
    int below2Count = 0;

    Image diff = image;
    for (int i = 0; i < image.getRows(); ++i) {
        for (int j = 0; j < image.getCols(); ++j) {
            if (denoisedImage(i, j) == image(i, j)) {
                diff(i, j) = 0.0;
				samePixelCount++;
            }
            else if (std::abs(denoisedImage(i, j) - image(i, j)) < 2.0) {
                diff(i, j) = 180.0;
				below2Count++;
            }
            else {
                diff(i, j) = 255.0;
            }
        }
    }

	std::cout << "Same pixels: " << samePixelCount 
              << ", Below 2 difference: " << below2Count 
              << ", Rest of the pixels: " << image.getRows() * image.getCols() - samePixelCount - below2Count << std::endl;

    cv::imshow("Difference", diff.toMat());
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
