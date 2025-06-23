#include <iostream>
#include <cmath>
#include <algorithm>
#include "Denoising.h"
#include "../Image/Image.h"

#include <fstream>

float tv_norm_and_grad(const Image& img, Image& grad, float eps) {
    const int rows = img.getRows();
    const int cols = img.getCols();
    float tv_norm = 0.0f;

    // Compute the total variation norm and gradient
    for (int i = 0; i < rows - 1; ++i) {
        for (int j = 0; j < cols - 1; ++j) {
            const float x_diff = img(i, j) - img(i, j + 1);
            const float y_diff = img(i, j) - img(i + 1, j);
            const float grad_mag = std::sqrt(x_diff * x_diff + y_diff * y_diff + eps);
            tv_norm += grad_mag;

            const float dx = x_diff / grad_mag;
            const float dy = y_diff / grad_mag;

            grad(i, j) += dx + dy;
            grad(i, j + 1) -= dx;
            grad(i + 1, j) -= dy;
        }
    }

    std::cout << "\tTV_Norm: " << tv_norm << std::endl;

    /*std::ofstream out("tv_norm_and_grad-grad-cpu.txt");
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            out << grad(i, j) << ' ';
        }
        out << std::endl;
    }
    out.close();*/

    return tv_norm;
}

float l2_norm_and_grad(const Image& img, const Image& orig, Image& grad) {
    const int rows = img.getRows();
    const int cols = img.getCols();
    float l2_norm = 0.0f;

    // Compute the L2 norm and gradient of the image and the original image
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            const float diff = img(i, j) - orig(i, j);
            grad(i, j) = diff;
            l2_norm += diff * diff;
        }
    }

    std::cout << "\tL2_Norm: " << 0.5 * l2_norm << std::endl;

    /*std::ofstream out("l2_norm_and_grad-grad-cpu.txt");
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            out << grad(i, j) << ' ';
        }
        out << std::endl;
    }
    out.close();*/

    return 0.5f * l2_norm;
}

float eval_loss_and_grad(const Image& img, const Image& orig, float strength, Image& grad) {
    const int rows = img.getRows();
    const int cols = img.getCols();
    Image tv_grad(rows, cols);
    const float tv_norm = tv_norm_and_grad(img, tv_grad);

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            grad(i, j) = strength * tv_grad(i, j);

    Image l2_grad(rows, cols);
    const float l2_norm = l2_norm_and_grad(img, orig, l2_grad);
    // Compute the combined weighted gradient
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            grad(i, j) += l2_grad(i, j);

    std::cout << "\tCombined loss: " << strength * tv_norm + l2_norm << std::endl;

    /*std::ofstream out("eval_loss_and_grad-grad-cpu.txt");
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            out << grad(i, j) << ' ';
        }
        out << std::endl;
    }
    out.close();*/

    // Compute the combined weighted loss
    return strength * tv_norm + l2_norm;
}

Image tv_denoise_gradient_descent(const Image& input, float strength, float step_size, float tol, bool suppress_log) {
    const int rows = input.getRows();
    const int cols = input.getCols();

    Image momentum(rows, cols);
    Image img = input;
    const Image orig_img = input;

    const float momentum_beta = 0.9f;
    const float loss_smoothing_beta = 0.9f;
    float loss_smoothed = 0.0f;

    const float step = step_size / (strength + 1);

    int counter = 1;
    while (true) {
        Image grad(rows, cols);
        float loss = eval_loss_and_grad(img, orig_img, strength, grad);

        if (!suppress_log) {
            std::cout << "Iteration: " << counter << ", Loss: " << loss << std::endl;
        }

        // Smooth the loss using exponential moving average
        // Smoothed loss is needed for more stable convergence
        loss_smoothed = loss_smoothed * loss_smoothing_beta + loss * (1.0f - loss_smoothing_beta);

        // Debias the smoothed loss to correct the bias introduced by the zero initialization
        float loss_smoothed_debiased = loss_smoothed / (1.0f - static_cast<float>(std::pow(loss_smoothing_beta, counter)));
        if (counter > 1 && loss_smoothed_debiased / loss < 1.0f + tol) {
            if (!suppress_log) {
                std::cout << "Converged after " << counter << " iterations with loss: " << loss_smoothed_debiased << std::endl;
            }
            break;
        }

        // Momentum keeps track of the previous gradients to stabilize and speed up convergence
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                momentum(i, j) *= momentum_beta;
                momentum(i, j) += grad(i, j) * (1.0f - momentum_beta);
                img(i, j) -= step / (1.0f - static_cast<float>(std::pow(momentum_beta, counter))) * momentum(i, j);
            }
        }

        std::cout << "\tLoss smoothed: " << loss_smoothed << std::endl;
        std::cout << "\tLoss smothed debiased: " << loss_smoothed_debiased << std::endl;

        /*std::ofstream out("tv_denoise_gradient_descent-momentum-cpu.txt");
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                out << momentum(i, j) << ' ';
            }
            out << std::endl;
        }
        out.close();

        std::ofstream out2("tv_denoise_gradient_descent-img-cpu.txt");
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                out2 << img(i, j) << ' ';
            }
            out2 << std::endl;
        }
        out2.close();*/

        /*if (counter == 1) {
            break;
        }*/

        /*if (counter == 50) {
            std::ofstream out2("tv_denoise_gradient_descent-img-step50-gpu.txt");
            for (int i = 0; i < img.getRows(); ++i) {
                for (int j = 0; j < img.getCols(); ++j) {
                    out2 << img(i, j) << ' ';
                }
                out2 << std::endl;
            }
            out2.close();
        }*/

        ++counter;
    }

    return img;
}
