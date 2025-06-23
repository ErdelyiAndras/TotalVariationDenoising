#define __NO_STD_VECTOR
#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <oclutils.hpp>
#include <string>
#include "../Image/Image.h"
#include "Denoising.h"

int main(int argc, char** argv) {
	if (argc != 6) {
		std::cerr << "Usage: " << argv[0] 
			      << " <input_image_path> <output_image_path> <strength> <step_size> <tol>" 
			      << std::endl;
		return -1;
	}

	try {
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

		float strength = std::stof(argv[3]);
		float step_size = std::stof(argv[4]);
		float tol = std::stof(argv[5]);

		auto start = std::chrono::high_resolution_clock::now();

		Image denoisedImage = tv_denoise_gradient_descent(context, queue, program, image, strength, step_size, tol);

		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<float> elapsed = end - start;
		std::cout << "GPU_Denoising took: " << elapsed.count() << " seconds" << std::endl;

		cv::Mat displayImage = denoisedImage.toMat();

		cv::imshow("Denoised", displayImage);
		cv::waitKey(0);

		std::string path = argv[2];
		cv::imwrite(path, displayImage);
	}
	catch (const std::exception& e) {
		std::cerr << "Exception: " << e.what() << std::endl;
		return -1;
	}
	catch (...) {
		std::cout << "Other exception" << std::endl;
		return -1;
	}
	
	return 0;
}
