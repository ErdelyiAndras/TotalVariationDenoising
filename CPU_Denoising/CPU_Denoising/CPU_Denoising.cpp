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

// grad has to be initialized with zeros
long double tv_norm_and_grad(const Image& img, Image& grad, long double eps = 1e-8) {
    const int rows = img.getRows();
    const int cols = img.getCols();
    long double tv_norm = 0.0L;

	// Compute the total variation norm and gradient
    for (int i = 0; i < rows - 1; ++i) {
        for (int j = 0; j < cols - 1; ++j) {
            const long double x_diff = img(i, j) - img(i, j + 1);
            const long double y_diff = img(i, j) - img(i + 1, j);
            const long double grad_mag = std::sqrt(x_diff * x_diff + y_diff * y_diff + eps);
            tv_norm += grad_mag;

            const long double dx = x_diff / grad_mag;
            const long double dy = y_diff / grad_mag;

            grad(i, j) = dx + dy;
            grad(i, j + 1) -= dx;
            grad(i + 1, j) -= dy;
        }
    }

    return tv_norm;
}

long double l2_norm_and_grad(const Image& img, const Image& orig, Image& grad) {
    const int rows = img.getRows();
    const int cols = img.getCols();
    long double l2_norm = 0.0L;

    // Compute the L2 norm and gradient of the image and the original image
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            const long double diff = img(i, j) - orig(i, j);
            grad(i, j) = diff;
            l2_norm += diff * diff;
        }
    }
    return 0.5L * l2_norm;
}

long double eval_loss_and_grad(const Image& img, const Image& orig, long double strength, Image& grad) {
	const int rows = img.getRows();
	const int cols = img.getCols();
    Image tv_grad(rows, cols);
    const long double tv_norm = tv_norm_and_grad(img, tv_grad);

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
			grad(i, j) = strength * tv_grad(i, j);

	Image l2_grad(rows, cols);
    const long double l2_norm = l2_norm_and_grad(img, orig, l2_grad);
    // Compute the combined weighted gradient
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            grad(i, j) += l2_grad(i, j);

    // Compute the combined weighted loss
	return strength * tv_norm + l2_norm;
}

Image tv_denoise_gradient_descent(const Image& input, long double strength, long double step_size = 1e-2, long double tol = 3.2e-3) {
    const int rows = input.getRows();
    const int cols = input.getCols();
    
    Image momentum(rows, cols);
    Image img = input;
    const Image orig_img = input;

    const long double momentum_beta = 0.9L;
    const long double loss_smoothing_beta = 0.9L;
    long double loss_smoothed = 0.0L;

	const long double step = step_size / (strength + 1);

    int counter = 1;
    while (true) {
        Image grad(rows, cols);
        long double loss = eval_loss_and_grad(img, orig_img, strength, grad);

		std::cout << "Iteration: " << counter << ", Loss: " << loss << std::endl;

		// Smooth the loss using exponential moving average
		// Smoothed loss is needed for more stable convergence
        loss_smoothed = loss_smoothed * loss_smoothing_beta + loss * (1.0L - loss_smoothing_beta);

        // Debias the smoothed loss to correct the bias introduced by the zero initialization
		long double loss_smoothed_debiased = loss_smoothed / (1.0L - std::pow(loss_smoothing_beta, counter));
        if (counter > 1 && loss_smoothed_debiased / loss < 1.0L + tol) {
            std::cout << "Converged after " << counter << " iterations with loss: " << loss_smoothed_debiased << std::endl;
            break;
		}
        
		// Momentum keeps track of the previous gradients to stabilize and speed up convergence
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                momentum(i, j) *= momentum_beta;
				momentum(i, j) += grad(i, j) * (1.0L - momentum_beta);
				img(i, j) -= step / (1 - std::pow(momentum_beta, counter)) * momentum(i, j);
            }
        }

        ++counter;
    }

    return img;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    Image image(argv[1]);

    Image denoisedImage = tv_denoise_gradient_descent(image, 0.1L);

    cv::Mat displayImage = denoisedImage.toMat();

    cv::imshow("Denoised", displayImage);
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
