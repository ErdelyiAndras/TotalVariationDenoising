#pragma once

#include <opencv2/opencv.hpp>
#include <string>

/**
 * @class Image
 * @brief A simple image class for handling 2D, single-channel (grayscale) images with long double precision.
 *
 * Provides constructors for creating images from dimensions, file paths, OpenCV matrices, or by copying another Image.
 * Supports element access, assignment, and conversion to OpenCV Mat format.
 */
class Image {
public:
	/**
	 * @brief Constructs an empty image or an image with the given dimensions.
	 * @param rows Number of rows (default: 0).
	 * @param cols Number of columns (default: 0).
	 */
	Image(int rows = 0, int cols = 0);

	/**
	 * @brief Constructs an image by loading from a file.
	 * @param path Path to the image file.
	 */
	Image(const std::string& path);

	/**
	 * @brief Constructs an image from an OpenCV matrix.
	 * @param mat OpenCV cv::Mat object.
	 */
	Image(const cv::Mat& mat);

	/**
	 * @brief Copy constructor.
	 * @param other Image to copy from.
	 */
	Image(const Image& other);

	/**
	 * @brief Destructor. Releases allocated memory.
	 */
	~Image();

	/**
	 * @brief Returns the number of rows in the image.
	 * @return Number of rows.
	 */
	inline int getRows() const { return rows; }

	/**
	 * @brief Returns the number of columns in the image.
	 * @return Number of columns.
	 */
	inline int getCols() const { return cols; }

	/**
	 * @brief Assignment operator.
	 * @param other Image to assign from.
	 * @return Reference to this image.
	 */
	Image operator=(const Image& other);

	/**
	 * @brief Accesses a pixel value (modifiable).
	 * @param row Row index.
	 * @param col Column index.
	 * @return Reference to the pixel value at (row, col).
	 */
	long double& operator()(int row, int col);

	/**
	 * @brief Accesses a pixel value (const).
	 * @param row Row index.
	 * @param col Column index.
	 * @return Const reference to the pixel value at (row, col).
	 */
	const long double& operator()(int row, int col) const;

	/**
	 * @brief Converts the image to an OpenCV cv::Mat object. Can be used to display or save the image, using OpenCV functions.
	 * @return cv::Mat representation of the image.
	 */
	cv::Mat toMat() const;

private:
	int rows;
	int cols;
	long double** image;
};
