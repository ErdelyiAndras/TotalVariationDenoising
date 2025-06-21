#define __NO_STD_VECTOR
#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <oclutils.hpp>
#include <string>
#include "../Image/Image.h"
#include "Denoising.h"

int main(int argc, char** argv)
{
	if (argc != 2) {
		std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
		return -1;
	}

	Image image(argv[1]);
	int img_size = image.getRows() * image.getCols();

	cl::Context context;
	if (!oclCreateContextBy(context, "intel")) {
		throw cl::Error(CL_INVALID_CONTEXT, "Failed to create a valid context!");
	}

	cl::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
	cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

	std::string source_code = oclReadSourcesFromFile("Denoising.cl");
	cl::Program::Sources sources(1, std::make_pair(source_code.c_str(), source_code.length() + 1));

	cl::Program program(context, sources);

	try {
		program.build(devices);
	}
	catch (cl::Error error) {
		oclPrintError(error);
		std::cerr << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) << std::endl;
		std::cerr << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices[0]) << std::endl;
		std::cerr << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
		return -1;
	}


	float* tv_norm_mtx = new float[img_size];
	float* dx_mtx = new float[img_size];
	float* dy_mtx = new float[img_size];

	tv_norm_mtx_and_dx_dy_mtx(context, queue, program, image, tv_norm_mtx, dx_mtx, dy_mtx);
	float tv_norm = sum<float>(context, queue, program, tv_norm_mtx, img_size);

	std::cout << tv_norm << std::endl;
}
