#include "Image.h"

Image::Image(const std::string& path) {
	cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
	if (img.empty()) {
		throw std::runtime_error("Failed to load image from path: " + path);
	}
	
	rows = img.rows;
	cols = img.cols;

	image = new long double* [rows];
	for (int i = 0; i < rows; ++i) {
		image[i] = new long double[cols];
		for (int j = 0; j < cols; ++j) {
			image[i][j] = static_cast<long double>(img.at<uchar>(i, j));
		}
	}
}

Image::Image(const Image& other) : image(other.image) {
	if (!other.image) {
		return;
	}
	
	rows = other.rows;
	cols = other.cols;
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			image[i][j] = other.image[i][j];
		}
	}
}

Image::~Image() {
	if (image) {
		for (int i = 0; i < rows; ++i) {
			delete[] image[i];
		}
		delete[] image;
		image = nullptr;
	}
}

long double& Image::operator()(int row, int col) {
	return image[row][col];
}

const long double& Image::operator()(int row, int col) const {
	return image[row][col];
}

cv::Mat Image::toMat() const {
	cv::Mat mat(rows, cols, CV_8U);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			mat.at<unsigned char>(i, j) = image[i][j];
		}
	}
	return mat;
}
