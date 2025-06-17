#pragma once

#include <opencv2/opencv.hpp>
#include <string>

class Image {
public:
	Image(int rows, int cols);
	Image(const std::string& path);
	Image(const cv::Mat& mat);
	Image(const Image& other);
	~Image();

	inline int getRows() const { return rows; }
	inline int getCols() const { return cols; }

	Image operator=(const Image& other);

	long double& operator()(int row, int col);
	const long double& operator()(int row, int col) const;

	cv::Mat toMat() const;

private:
	int rows;
	int cols;
	long double** image;
};
