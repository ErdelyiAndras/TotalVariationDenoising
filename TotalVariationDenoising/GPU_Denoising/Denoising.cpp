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
	cl::Context& context, cl::CommandQueue& queue, cl::Program& program, 
	const Image& image, float* tv_norm_mtx, float* dx_mtx, float* dy_mtx, float eps
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
	kernel.setArg(6, eps);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, img_size, cl::NullRange);

	queue.enqueueReadBuffer(tv_norm_mtx_buffer, CL_TRUE, 0, img_size * sizeof(float), tv_norm_mtx);
	queue.enqueueReadBuffer(dx_mtx_buffer, CL_TRUE, 0, img_size * sizeof(float), dx_mtx);
	queue.enqueueReadBuffer(dy_mtx_buffer, CL_TRUE, 0, img_size * sizeof(float), dy_mtx);
}

void grad_from_dx_dy_mtxs(
	cl::Context& context, cl::CommandQueue& queue, cl::Program& program,
	const float* dx_mtx, const float* dy_mtx, float* grad, int rows, int cols
) {
	const int img_size = rows * cols;

	cl::Buffer dx_mtx_buffer(context, CL_MEM_READ_WRITE, img_size * sizeof(float));
	queue.enqueueWriteBuffer(dx_mtx_buffer, CL_TRUE, 0, img_size * sizeof(float), dx_mtx);
	queue.finish();

	cl::Buffer dy_mtx_buffer(context, CL_MEM_READ_WRITE, img_size * sizeof(float));
	queue.enqueueWriteBuffer(dy_mtx_buffer, CL_TRUE, 0, img_size * sizeof(float), dy_mtx);
	queue.finish();

	cl::Buffer grad_buffer(context, CL_MEM_READ_WRITE, img_size * sizeof(float));
	queue.enqueueWriteBuffer(grad_buffer, CL_TRUE, 0, img_size * sizeof(float), std::vector<float>(img_size, 0.0f).data());
	queue.finish();

	for (int i = 1; i <= 3; ++i) {
		std::string kernel_name = "grad_from_dx_dy_step" + std::to_string(i);
		cl::Kernel kernel(program, kernel_name.c_str());

		kernel.setArg(0, dx_mtx_buffer);
		kernel.setArg(1, dy_mtx_buffer);
		kernel.setArg(2, grad_buffer);
		kernel.setArg(3, rows);
		kernel.setArg(4, cols);

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, img_size, cl::NullRange);
		queue.finish();
	}
	queue.enqueueReadBuffer(grad_buffer, CL_TRUE, 0, img_size * sizeof(float), grad);
}

float tv_norm_and_grad(
	cl::Context& context, cl::CommandQueue& queue, cl::Program& program,
	const Image& image, float* grad, float eps
) {
	const int img_size = image.getRows() * image.getCols();

	float* tv_norm_mtx = new float[img_size];
	float* dx_mtx = new float[img_size];
	float* dy_mtx = new float[img_size];

	tv_norm_mtx_and_dx_dy_mtx(context, queue, program, image, tv_norm_mtx, dx_mtx, dy_mtx, eps);
	float tv_norm = sum<float>(context, queue, program, tv_norm_mtx, img_size);

	grad_from_dx_dy_mtxs(context, queue, program, dx_mtx, dy_mtx, grad, image.getRows(), image.getCols());

	delete[] tv_norm_mtx;
	delete[] dx_mtx;
	delete[] dy_mtx;

	return tv_norm;
}

void l2_norm_mtx_and_grad(
	cl::Context& context, cl::CommandQueue& queue, cl::Program& program,
	const Image& img, const Image& orig, float* l2_norm_mtx, float* grad
) {
	const int img_size = img.getRows() * img.getCols();

	cl::Kernel kernel(program, "l2_norm_mtx_and_grad");

	cl::Buffer img_buffer(context, CL_MEM_READ_WRITE, img_size * sizeof(float));
	queue.enqueueWriteBuffer(img_buffer, CL_TRUE, 0, img_size * sizeof(float), img.data());
	queue.finish();

	cl::Buffer orig_buffer(context, CL_MEM_READ_WRITE, img_size * sizeof(float));
	queue.enqueueWriteBuffer(orig_buffer, CL_TRUE, 0, img_size * sizeof(float), orig.data());
	queue.finish();

	cl::Buffer l2_norm_mtx_buffer(context, CL_MEM_READ_WRITE, img_size * sizeof(float));
	queue.enqueueWriteBuffer(l2_norm_mtx_buffer, CL_TRUE, 0, img_size * sizeof(float), l2_norm_mtx);
	queue.finish();

	cl::Buffer grad_buffer(context, CL_MEM_READ_WRITE, img_size * sizeof(float));
	queue.enqueueWriteBuffer(grad_buffer, CL_TRUE, 0, img_size * sizeof(float), grad);
	queue.finish();

	kernel.setArg(0, img_buffer);
	kernel.setArg(1, orig_buffer);
	kernel.setArg(2, l2_norm_mtx_buffer);
	kernel.setArg(3, grad_buffer);
	kernel.setArg(4, img.getRows());
	kernel.setArg(5, img.getCols());

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, img_size, cl::NullRange);

	queue.enqueueReadBuffer(l2_norm_mtx_buffer, CL_TRUE, 0, img_size * sizeof(float), l2_norm_mtx);
	queue.enqueueReadBuffer(grad_buffer, CL_TRUE, 0, img_size * sizeof(float), grad);
}

float l2_norm_and_grad(
	cl::Context& context, cl::CommandQueue& queue, cl::Program& program,
	const Image& img, const Image& orig, float* grad
) {
	const int img_size = img.getRows() * img.getCols();

	float* l2_norm_mtx = new float[img_size];

	l2_norm_mtx_and_grad(context, queue, program, img, orig, l2_norm_mtx, grad);

	float l2_norm = sum<float>(context, queue, program, l2_norm_mtx, img_size);

	delete[] l2_norm_mtx;

	return l2_norm;
}