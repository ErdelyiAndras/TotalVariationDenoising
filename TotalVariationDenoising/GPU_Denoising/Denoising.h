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
T sum(cl::Context& context, cl::CommandQueue& queue, cl::Program& program, T* array, int size) {
	if (size == 0) {
		return static_cast<T>(0);
	}

	cl::Kernel kernel = init_sum_kernel<T>(program);

	cl::Buffer array_buffer(context, CL_MEM_READ_WRITE, size * sizeof(T));
	queue.enqueueWriteBuffer(array_buffer, CL_TRUE, 0, size * sizeof(T), array);
	queue.finish();

	kernel.setArg(0, array_buffer);

	int extended_size = 1;
	while (extended_size < size) {
		extended_size *= 2;
	}


	for (int offset = extended_size / 2; offset > 0; offset >>= 1) {
		kernel.setArg(1, offset);

		if (offset == extended_size / 2) {
			kernel.setArg(2, size);
		}
		else {
			kernel.setArg(2, offset * 2);
		}

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, offset, cl::NullRange);
	}

	T result;
	queue.enqueueReadBuffer(array_buffer, CL_TRUE, 0, sizeof(T), &result);

	return result;
}

void tv_norm_mtx_and_dx_dy_mtx(
	cl::Context& context, cl::CommandQueue& queue, cl::Program& program, const Image& image,
	float* tv_norm_mtx, float* dx_mtx, float* dy_mtx
);

