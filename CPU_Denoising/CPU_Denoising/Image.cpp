#include "Image.h"

Image::Image(int rows = 0, int cols = 0) : rows(rows), cols(cols), image(nullptr) {
	if (rows < 0 || cols < 0) {
		throw std::invalid_argument("Rows and columns must be non-negative.");
	}
	if (rows == 0 || cols == 0) {
		return;
	}
	image = new long double* [rows];
	for (int i = 0; i < rows; ++i) {
		image[i] = new long double[cols];
		for (int j = 0; j < cols; ++j) {
			image[i][j] = 0.0L; // Initialize with zero
		}
	}
}

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
			image[i][j] = static_cast<long double>(img.at<unsigned char>(i, j));
		}
	}
}

Image::Image(const cv::Mat& mat) {
	if (mat.empty() || mat.type() != CV_8U) {
		throw std::runtime_error("Invalid image matrix provided.");
	}

	rows = mat.rows;
	cols = mat.cols;

	image = new long double* [rows];
	for (int i = 0; i < rows; ++i) {
		image[i] = new long double[cols];
		for (int j = 0; j < cols; ++j) {
			image[i][j] = static_cast<long double>(mat.at<unsigned char>(i, j));
		}
	}
}

Image::Image(const Image& other) : rows(0), cols(0), image(nullptr) {
	if (!other.image) {
		return;
	}
	
	rows = other.rows;
	cols = other.cols;
	image = new long double* [rows];
	for (int i = 0; i < rows; ++i) {
		image[i] = new long double[cols];
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

Image Image::operator=(const Image& other) {
	if (this == &other) {
		return *this;
	}

	if (image) {
		for (int i = 0; i < rows; ++i) {
			delete[] image[i];
		}
		delete[] image;
	}

	rows = other.rows;
	cols = other.cols;

	image = new long double* [rows];
	for (int i = 0; i < rows; ++i) {
		image[i] = new long double[cols];
		for (int j = 0; j < cols; ++j) {
			image[i][j] = other.image[i][j];
		}
	}
	return *this;
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
			mat.at<unsigned char>(i, j) = static_cast<unsigned char>(std::max(std::min(image[i][j], 255.0L), 0.0L));
		}
	}
	return mat;
}
