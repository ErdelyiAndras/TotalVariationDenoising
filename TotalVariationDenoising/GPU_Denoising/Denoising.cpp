#include <CL/cl.hpp>
#include <vector>
#include "Denoising.h"
#include "../Image/Image.h"

#include <fstream>

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
	/*float tv_norm = 0.0f;
	for (int i = 0; i < img_size; ++i) {
		tv_norm += tv_norm_mtx[i];
	}*/

	grad_from_dx_dy_mtxs(context, queue, program, dx_mtx, dy_mtx, grad, image.getRows(), image.getCols());

	delete[] tv_norm_mtx;
	delete[] dx_mtx;
	delete[] dy_mtx;

	std::cout << "\tTV_Norm: " << tv_norm << std::endl;

	/*std::ofstream out("tv_norm_and_grad-grad-gpu.txt");
	for (int i = 0; i < image.getRows(); ++i) {
		for (int j = 0; j < image.getCols(); ++j) {
			out << grad[i * image.getCols() + j] << ' ';
		}
		out << std::endl;
	}
	out.close();*/

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
	/*float l2_norm = 0.0f;
	for (int i = 0; i < img_size; ++i) {
		l2_norm += l2_norm_mtx[i];
	}*/

	delete[] l2_norm_mtx;

	std::cout << "\tL2_Norm: " << 0.5 * l2_norm << std::endl;

	/*std::ofstream out("l2_norm_and_grad-grad-gpu.txt");
	for (int i = 0; i < img.getRows(); ++i) {
		for (int j = 0; j < img.getCols(); ++j) {
			out << grad[i * img.getCols() + j] << ' ';
		}
		out << std::endl;
	}
	out.close();*/

	return l2_norm;
}

float eval_loss_and_grad(
	cl::Context& context, cl::CommandQueue& queue, cl::Program& program,
	const Image& img, const Image& orig, float strength, float* grad
) {
	const int img_size = img.getRows() * img.getCols();

	float* tv_grad = new float[img_size];
	const float tv_norm = tv_norm_and_grad(context, queue, program, img, tv_grad);

	cl::Kernel kernel(program, "eval_loss_and_grad");

	cl::Buffer grad_buffer(context, CL_MEM_READ_WRITE, img_size * sizeof(float));
	queue.enqueueWriteBuffer(grad_buffer, CL_TRUE, 0, img_size * sizeof(float), std::vector<float>(img_size, 0.0f).data());
	queue.finish();
	
	cl::Buffer tv_grad_buffer(context, CL_MEM_READ_WRITE, img_size * sizeof(float));
	queue.enqueueWriteBuffer(tv_grad_buffer, CL_TRUE, 0, img_size * sizeof(float), tv_grad);
	queue.finish();

	kernel.setArg(0, grad_buffer);
	kernel.setArg(1, tv_grad_buffer);
	kernel.setArg(2, strength);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, img_size, cl::NullRange);
	
	float* l2_grad = new float[img_size];
	const float l2_norm = l2_norm_and_grad(context, queue, program, img, orig, l2_grad);

	cl::Buffer l2_grad_buffer(context, CL_MEM_READ_WRITE, img_size * sizeof(float));
	queue.enqueueWriteBuffer(l2_grad_buffer, CL_TRUE, 0, img_size * sizeof(float), l2_grad);
	queue.finish();

	kernel.setArg(1, l2_grad_buffer);
	kernel.setArg(2, 1.0f);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, img_size, cl::NullRange);

	queue.enqueueReadBuffer(grad_buffer, CL_TRUE, 0, img_size * sizeof(float), grad);

	delete[] tv_grad;
	delete[] l2_grad;
	
	std::cout << "\tCombined loss: " << strength * tv_norm + l2_norm << std::endl;

	/*std::ofstream out("eval_loss_and_grad-grad-gpu.txt");
	for (int i = 0; i < img.getRows(); ++i) {
		for (int j = 0; j < img.getCols(); ++j) {
			out << grad[i * img.getCols() + j] << ' ';
		}
		out << std::endl;
	}
	out.close();*/

	return strength * tv_norm + l2_norm;
}


void eval_momentum(
	cl::Context& context, cl::CommandQueue& queue, cl::Program& program,
	float* momentum, const float* grad, float momentum_beta, int img_size
) {
	cl::Kernel momentum_kernel(program, "eval_momentum");

	cl::Buffer momentum_buffer(context, CL_MEM_READ_WRITE, img_size * sizeof(float));
	queue.enqueueWriteBuffer(momentum_buffer, CL_TRUE, 0, img_size * sizeof(float), momentum);
	queue.finish();

	cl::Buffer grad_buffer(context, CL_MEM_READ_WRITE, img_size * sizeof(float));
	queue.enqueueWriteBuffer(grad_buffer, CL_TRUE, 0, img_size * sizeof(float), grad);
	queue.finish();

	momentum_kernel.setArg(0, momentum_buffer);
	momentum_kernel.setArg(1, grad_buffer);
	momentum_kernel.setArg(2, momentum_beta);

	queue.enqueueNDRangeKernel(momentum_kernel, cl::NullRange, img_size, cl::NullRange);

	queue.enqueueReadBuffer(momentum_buffer, CL_TRUE, 0, img_size * sizeof(float), momentum);
}

void update_img(
	cl::Context& context, cl::CommandQueue& queue, cl::Program& program,
	float* img, const float* momentum, int img_size, float step, float momentum_beta, int counter
) {
	cl::Kernel update_kernel(program, "update_img");

	cl::Buffer img_buffer(context, CL_MEM_READ_WRITE, img_size * sizeof(float));
	queue.enqueueWriteBuffer(img_buffer, CL_TRUE, 0, img_size * sizeof(float), img);
	queue.finish();

	cl::Buffer momentum_buffer2(context, CL_MEM_READ_WRITE, img_size * sizeof(float));
	queue.enqueueWriteBuffer(momentum_buffer2, CL_TRUE, 0, img_size * sizeof(float), momentum);
	queue.finish();

	update_kernel.setArg(0, img_buffer);
	update_kernel.setArg(1, momentum_buffer2);
	update_kernel.setArg(2, step);
	update_kernel.setArg(3, momentum_beta);
	update_kernel.setArg(4, counter);

	queue.enqueueNDRangeKernel(update_kernel, cl::NullRange, img_size, cl::NullRange);

	queue.enqueueReadBuffer(img_buffer, CL_TRUE, 0, img_size * sizeof(float), img);
}

Image tv_denoise_gradient_descent(
	cl::Context& context, cl::CommandQueue& queue, cl::Program& program,
	const Image& input, float strength, float step_size, float tol, bool suppress_log
) {
	const int img_size = input.getRows() * input.getCols();

	std::vector<float> momentum_vector(img_size, 0.0f);
	float* momentum = momentum_vector.data();

	Image img = input;
	const Image orig_img = input;

	const float momentum_beta = 0.9f;
	const float loss_smoothing_beta = 0.9f;
	float loss_smoothed = 0.0f;

	const float step = step_size / (strength + 1);

	int counter = 1;
	while (true) {
		float* grad = new float[img_size];
		float loss = eval_loss_and_grad(context, queue, program, img, orig_img, strength, grad);

		if (!suppress_log) {
			std::cout << "Iteration: " << counter << ", Loss: " << loss << std::endl;
		}

		loss_smoothed = loss_smoothed * loss_smoothing_beta + loss * (1.0f - loss_smoothing_beta);

		float loss_smoothed_debiased = loss_smoothed / (1.0f - static_cast<float>(std::pow(loss_smoothing_beta, counter)));
		if (counter > 1 && loss_smoothed_debiased / loss < 1.0f + tol) {
			if (!suppress_log) {
				std::cout << "Converged after " << counter << " iterations with loss: " << loss_smoothed_debiased << std::endl;
			}
			break;
		}

		eval_momentum(context, queue, program, momentum, grad, momentum_beta, img_size);
		update_img(context, queue, program, img.data(), momentum, img_size, step, momentum_beta, counter);
		/*for (int i = 0; i < img.getRows(); ++i) {
			for (int j = 0; j < img.getCols(); ++j) {
				momentum[i * img.getCols() + j] *= momentum_beta;
				momentum[i * img.getCols() + j] += grad[i * img.getCols() + j] * (1.0f - momentum_beta);
				img(i, j) -= step / (1.0f - static_cast<float>(std::pow(momentum_beta, counter))) * momentum[i * img.getCols() + j];
			}
		}*/


		std::cout << "\tLoss smoothed: " << loss_smoothed << std::endl;
		std::cout << "\tLoss smothed debiased: " << loss_smoothed_debiased << std::endl << std::endl;

		/*std::ofstream out("tv_denoise_gradient_descent-momentum-gpu.txt");
		for (int i = 0; i < img.getRows(); ++i) {
			for (int j = 0; j < img.getCols(); ++j) {
				out << momentum[i * img.getCols() + j] << ' ';
			}
			out << std::endl;
		}
		out.close();

		std::ofstream out2("tv_denoise_gradient_descent-img-gpu.txt");
		for (int i = 0; i < img.getRows(); ++i) {
			for (int j = 0; j < img.getCols(); ++j) {
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

		delete[] grad;
	}

	return img;
}
