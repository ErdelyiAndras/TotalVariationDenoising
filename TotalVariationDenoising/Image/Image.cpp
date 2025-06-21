#include "pch.h"

#include "Image.h"

Image::Image(int rows, int cols) : rows(rows), cols(cols), image(nullptr) {
	if (rows < 0 || cols < 0) {
		throw std::invalid_argument("Rows and columns must be non-negative.");
	}
	if (rows == 0 || cols == 0) {
		return;
	}
	image = new float[rows * cols];
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			this->operator()(i, j) = 0.0f;
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

	image = new float[rows * cols];
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			this->operator()(i, j) = static_cast<float>(img.at<unsigned char>(i, j)) / 255.0f;
		}
	}
}

Image::Image(const cv::Mat& mat) {
	if (mat.empty() || mat.type() != CV_8U) {
		throw std::runtime_error("Invalid image matrix provided.");
	}

	rows = mat.rows;
	cols = mat.cols;

	image = new float[rows * cols];
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			this->operator()(i, j) = static_cast<float>(mat.at<unsigned char>(i, j));
		}
	}
}

Image::Image(const Image& other) : rows(0), cols(0), image(nullptr) {
	if (!other.image) {
		return;
	}
	
	rows = other.rows;
	cols = other.cols;
	image = new float[rows * cols];
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			this->operator()(i, j) = other.image[i * cols + j];
		}
	}
}

Image::~Image() {
	if (image) {
		delete[] image;
		image = nullptr;
	}
}

Image Image::operator=(const Image& other) {
	if (this == &other) {
		return *this;
	}

	if (image) {
		delete[] image;
		image = nullptr;
	}

	rows = other.rows;
	cols = other.cols;

	image = new float[rows * cols];
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			this->operator()(i, j) = other.image[i * cols + j];
		}
	}
	return *this;
}

float& Image::operator()(int row, int col) {
	return image[row * cols + col];
}

const float& Image::operator()(int row, int col) const {
	return image[row * cols + col];
}

cv::Mat Image::toMat() const {
	cv::Mat mat(rows, cols, CV_8U);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			mat.at<unsigned char>(i, j) = static_cast<unsigned char>(std::max(std::min(this->operator()(i, j) * 255.0f, 255.0f), 0.0f));
		}
	}
	return mat;
}
