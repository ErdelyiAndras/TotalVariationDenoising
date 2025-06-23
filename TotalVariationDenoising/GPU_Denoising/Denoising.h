#pragma once

#include <CL/cl.hpp>
#include <string>
#include <typeinfo>
#include "../Image/Image.h"

/**
 * @brief Initializes the sum reduction kernel for the given type.
 * @tparam T Data type (only int and float are supported).
 * @param program Compiled OpenCL program containing the sum kernel.
 * @return The OpenCL kernel object for sum reduction.
 * @throws std::runtime_error if the type is not supported.
 */
template <typename T>
cl::Kernel init_sum_kernel(cl::Program& program) {
	throw std::runtime_error(std::string("Unsupported type: ") + typeid(T).name());
}

/**
 * @brief Specialization for int type.
 */
template <>
cl::Kernel init_sum_kernel<int>(cl::Program& program);

/**
 * @brief Specialization for float type.
 */
template <>
cl::Kernel init_sum_kernel<float>(cl::Program& program);

/**
 * @brief Performs parallel sum reduction on the GPU for the given array.
 * @tparam T Data type (int or float).
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param program Compiled OpenCL program containing the sum kernel.
 * @param array Pointer to the input array.
 * @param size Number of elements in the array.
 * @return The sum of all elements in the array.
 */
template <typename T>
T sum(cl::Context& context, cl::CommandQueue& queue, cl::Program& program, const T* array, int size) {
	if (size == 0) {
		return static_cast<T>(0);
	}

	cl::Kernel kernel = init_sum_kernel<T>(program);

	int extended_size = 1;
	while (extended_size < size) {
		extended_size *= 2;
	}

	std::vector<T> padded_array(extended_size, static_cast<T>(0));
	std::copy(array, array + size, padded_array.begin());

	cl::Buffer array_buffer(context, CL_MEM_READ_WRITE, extended_size * sizeof(T));
	queue.enqueueWriteBuffer(array_buffer, CL_TRUE, 0, extended_size * sizeof(T), padded_array.data());
	queue.finish();

	kernel.setArg(0, array_buffer);

	for (int offset = extended_size / 2; offset > 0; offset >>= 1) {
		kernel.setArg(1, offset);

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, offset, cl::NullRange);
		queue.finish();
	}

	T result;
	queue.enqueueReadBuffer(array_buffer, CL_TRUE, 0, sizeof(T), &result);

	return result;
}


/**
 * @brief Computes the TV norm matrix and the dx, dy matrices for an image on the GPU.
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param program Compiled OpenCL program.
 * @param image Input image.
 * @param tv_norm_mtx Output buffer for TV norm contributions (size: rows * cols).
 * @param dx_mtx Output buffer for dx values (size: rows * cols).
 * @param dy_mtx Output buffer for dy values (size: rows * cols).
 * @param eps Small epsilon value to avoid division by zero.
 */
void tv_norm_mtx_and_dx_dy_mtx(
	cl::Context& context, cl::CommandQueue& queue, cl::Program& program, 
	const Image& image, float* tv_norm_mtx, float* dx_mtx, float* dy_mtx, float eps
);

/**
 * @brief Computes the TV gradient from dx and dy matrices on the GPU.
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param program Compiled OpenCL program.
 * @param dx_mtx Input dx matrix.
 * @param dy_mtx Input dy matrix.
 * @param grad Output gradient buffer (size: rows * cols).
 * @param rows Number of rows in the image.
 * @param cols Number of columns in the image.
 */
void grad_from_dx_dy_mtxs(
	cl::Context& context, cl::CommandQueue& queue, cl::Program& program,
	const float* dx_mtx, const float* dy_mtx, float* grad, int rows, int cols
);

/**
 * @brief Computes the total variation norm and its gradient for an image on the GPU.
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param program Compiled OpenCL program.
 * @param image Input image.
 * @param grad Output gradient buffer (size: rows * cols).
 * @param eps Small epsilon value to avoid division by zero (default: 1e-8f).
 * @return The total variation norm.
 */
float tv_norm_and_grad(
	cl::Context& context, cl::CommandQueue& queue, cl::Program& program, 
	const Image& image, float* grad, float eps = 1e-8f
);

/**
 * @brief Computes the L2 norm matrix and its gradient between two images on the GPU.
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param program Compiled OpenCL program.
 * @param img Input image.
 * @param orig Original/reference image.
 * @param l2_norm_mtx Output buffer for L2 norm contributions (size: rows * cols).
 * @param grad Output gradient buffer (size: rows * cols).
 */
void l2_norm_mtx_and_grad(
	cl::Context& context, cl::CommandQueue& queue, cl::Program& program,
	const Image& img, const Image& orig, float* l2_norm_mtx, float* grad
);

/**
 * @brief Computes the L2 norm and its gradient between two images on the GPU.
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param program Compiled OpenCL program.
 * @param img Input image.
 * @param orig Original/reference image.
 * @param grad Output gradient buffer (size: rows * cols).
 * @return The L2 norm.
 */
float l2_norm_and_grad(
	cl::Context& context, cl::CommandQueue& queue, cl::Program& program,
	const Image& img, const Image& orig, float* grad
);


/**
 * @brief Computes the total loss (TV + L2) and its gradient for an image on the GPU.
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param program Compiled OpenCL program.
 * @param img Input image.
 * @param orig Original/reference image.
 * @param strength Weight for the TV loss term.
 * @param grad Output gradient buffer (size: rows * cols).
 * @return The total loss.
 */
float eval_loss_and_grad(
	cl::Context& context, cl::CommandQueue& queue, cl::Program& program,
	const Image& img, const Image& orig, float strength, float* grad
);


/**
 * @brief Updates the momentum buffer using the current gradient on the GPU.
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param program Compiled OpenCL program.
 * @param momentum Input/output momentum buffer (size: rows * cols).
 * @param grad Input gradient buffer (size: rows * cols).
 * @param momentum_beta Momentum weight parameter.
 * @param img_size Number of pixels in the image.
 */
void eval_momentum(
	cl::Context& context, cl::CommandQueue& queue, cl::Program& program,
	float* momentum, const float* grad, float momentum_beta, int img_size
);

/**
 * @brief Updates the image using the current momentum buffer on the GPU.
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param program Compiled OpenCL program.
 * @param img Input/output image buffer (size: rows * cols).
 * @param momentum Input momentum buffer (size: rows * cols).
 * @param img_size Number of pixel in the image.
 * @param step Step size (learning rate).
 * @param momentum_beta Momentum weight parameter.
 * @param counter Iteration counter (for bias correction).
 */
void update_img(
	cl::Context& context, cl::CommandQueue& queue, cl::Program& program,
	float* img, const float* momentum, int img_size, float step, float momentum_beta, int counter
);

/**
 * @brief Performs total variation denoising using gradient descent on the GPU.
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param program Compiled OpenCL program.
 * @param input Noisy input image.
 * @param strength Weight for the TV loss term.
 * @param step_size Step size (learning rate) for gradient descent (default: 1e-2f).
 * @param tol Tolerance for convergence (default: 3.2e-3f).
 * @param suppress_log If true, suppresses logging output (default: true).
 * @return The denoised image.
 */
Image tv_denoise_gradient_descent(
	cl::Context& context, cl::CommandQueue& queue, cl::Program& program,
	const Image& input, float strength, float step_size = 1e-2f, float tol = 3.2e-3f, bool suppress_log = true
);
