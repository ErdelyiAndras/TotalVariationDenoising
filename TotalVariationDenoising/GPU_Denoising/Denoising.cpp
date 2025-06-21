#include <CL/cl.hpp>
#include <vector>
#include "Denoising.h"
#include "../Image/Image.h"

template <>
cl::Kernel init_sum_kernel<int>(cl::Program& program) {
	return cl::Kernel(program, "sum_int");
}

template <>
cl::Kernel init_sum_kernel<float>(cl::Program& program) {
	return cl::Kernel(program, "sum_float");
}

void tv_norm_mtx_and_dx_dy_mtx(
	cl::Context& context, cl::CommandQueue& queue, cl::Program& program, const Image& image,
	float* tv_norm_mtx, float* dx_mtx, float* dy_mtx
) {
	const int img_size = image.getRows() * image.getCols();

	cl::Kernel kernel(program, "tv_norm_mtx_and_dx_dy");

	cl::Buffer img_buffer(context, CL_MEM_READ_WRITE, img_size * sizeof(float));
	queue.enqueueWriteBuffer(img_buffer, CL_TRUE, 0, img_size * sizeof(float), image.data());
	queue.finish();

	cl::Buffer tv_norm_mtx_buffer(context, CL_MEM_READ_WRITE, img_size * sizeof(float));
	queue.enqueueWriteBuffer(tv_norm_mtx_buffer, CL_TRUE, 0, img_size * sizeof(float), std::vector<float>(img_size, 0.0f).data());
	queue.finish();

	cl::Buffer dx_mtx_buffer(context, CL_MEM_READ_WRITE, img_size * sizeof(float));
	queue.enqueueWriteBuffer(dx_mtx_buffer, CL_TRUE, 0, img_size * sizeof(float), std::vector<float>(img_size, 0.0f).data());
	queue.finish();

	cl::Buffer dy_mtx_buffer(context, CL_MEM_READ_WRITE, img_size * sizeof(float));
	queue.enqueueWriteBuffer(dy_mtx_buffer, CL_TRUE, 0, img_size * sizeof(float), std::vector<float>(img_size, 0.0f).data());
	queue.finish();

	kernel.setArg(0, img_buffer);
	kernel.setArg(1, tv_norm_mtx_buffer);
	kernel.setArg(2, dx_mtx_buffer);
	kernel.setArg(3, dy_mtx_buffer);
	kernel.setArg(4, image.getRows());
	kernel.setArg(5, image.getCols());
	kernel.setArg(6, 1e-8f);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, img_size, cl::NullRange);

	queue.enqueueReadBuffer(tv_norm_mtx_buffer, CL_TRUE, 0, img_size * sizeof(float), tv_norm_mtx);
	queue.enqueueReadBuffer(dx_mtx_buffer, CL_TRUE, 0, img_size * sizeof(float), dx_mtx);
	queue.enqueueReadBuffer(dy_mtx_buffer, CL_TRUE, 0, img_size * sizeof(float), dy_mtx);
}