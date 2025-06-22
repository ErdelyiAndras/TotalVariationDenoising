#pragma once

#include <CL/cl.hpp>
#include <string>
#include <typeinfo>
#include "../Image/Image.h"

template <typename T>
cl::Kernel init_sum_kernel(cl::Program& program) {
	throw std::runtime_error(std::string("Unsupported type: ") + typeid(T).name());
}

template <>
cl::Kernel init_sum_kernel<int>(cl::Program& program);

template <>
cl::Kernel init_sum_kernel<float>(cl::Program& program);

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


void tv_norm_mtx_and_dx_dy_mtx(
	cl::Context& context, cl::CommandQueue& queue, cl::Program& program, 
	const Image& image, float* tv_norm_mtx, float* dx_mtx, float* dy_mtx, float eps
);

void grad_from_dx_dy_mtxs(
	cl::Context& context, cl::CommandQueue& queue, cl::Program& program,
	const float* dx_mtx, const float* dy_mtx, float* grad, int rows, int cols
);

float tv_norm_and_grad(
	cl::Context& context, cl::CommandQueue& queue, cl::Program& program, 
	const Image& image, float* grad, float eps = 1e-8f
);


void l2_norm_mtx_and_grad(
	cl::Context& context, cl::CommandQueue& queue, cl::Program& program,
	const Image& img, const Image& orig, float* l2_norm_mtx, float* grad
);

float l2_norm_and_grad(
	cl::Context& context, cl::CommandQueue& queue, cl::Program& program,
	const Image& img, const Image& orig, float* grad
);


float eval_loss_and_grad(
	cl::Context& context, cl::CommandQueue& queue, cl::Program& program,
	const Image& img, const Image& orig, float strength, float* grad
);


void eval_momentum(
	cl::Context& context, cl::CommandQueue& queue, cl::Program& program,
	float* momentum, const float* grad, float strength, int img_size
);

void update_img(
	cl::Context& context, cl::CommandQueue& queue, cl::Program& program,
	float* img, const float* momentum, int img_size, float step, float momentum_beta, int counter
);

Image tv_denoise_gradient_descent(
	cl::Context& context, cl::CommandQueue& queue, cl::Program& program,
	const Image& input, float strength, float step_size = 1e-2f, float tol = 3.2e-3f, bool suppress_log = true
);