#pragma once

#include "Image.h"

/**
 * @brief Computes the total variation (TV) norm of an image and its gradient.
 *
 * @param img Input image.
 * @param grad Output image to store the gradient of the TV norm (modified in-place). All values has to be set to zero before calling this function.
 * @param eps Small value to avoid division by zero (default: 1e-8).
 * @return The total variation norm as a long double.
 */
long double tv_norm_and_grad(const Image& img, Image& grad, long double eps = 1e-8);

/**
 * @brief Computes the L2 norm (squared error) between two images and its gradient.
 *
 * @param img Denoised image (input).
 * @param orig Original image (reference).
 * @param grad Output image to store the gradient of the L2 loss (modified in-place).
 * @return The L2 norm as a long double.
 */
long double l2_norm_and_grad(const Image& img, const Image& orig, Image& grad);

/**
 * @brief Computes the total loss (TV + L2) and its gradient for an image.
 *
 * @param img Denoised image (input).
 * @param orig Original image (reference).
 * @param strength Weight for the TV loss term.
 * @param grad Output image to store the combined gradient (modified in-place).
 * @return The total loss as a long double.
 */
long double eval_loss_and_grad(const Image& img, const Image& orig, long double strength, Image& grad);

/**
 * @brief Performs total variation denoising using gradient descent.
 *
 * Minimizes a loss function combining TV and L2 loss using momentum-based gradient descent.
 *
 * @param input Noisy input image.
 * @param strength Weight for the TV loss term.
 * @param step_size Step size (learning rate) for gradient descent (default: 1e-2).
 * @param tol Tolerance for convergence (default: 3.2e-3).
 * @return The denoised image.
 */
Image tv_denoise_gradient_descent(const Image& input, long double strength, long double step_size = 1e-2, long double tol = 3.2e-3);
