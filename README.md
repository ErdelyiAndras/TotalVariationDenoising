# Total Variation Denoising

This is an image denoising application that uses Total Variation Denoising and runs on the GPU. The repository includes the Visual Studio C++/OpenCL project with implementation for both the CPU and the GPU, a simple GUI script written in Python using TKinter and a noisy image generator script also written in Python.

## Prerequisites

- Python 3.x (for GUI and noisy image generator)
- Required Python packages: Numpy, TKinter, Pillow
- Visual Studio / MSBuild to build the C++ project
- OpenCV, OpenCL for C++ code

## Steps to Run the Application

### 0. Change Directory to the Project Root

Navigate to the project root directory:

```sh
cd path/to/this/project
```

### 1. Generate a Noisy Image

You can use the noisy image generator script or use your own noisy images:

```sh
py .\noisy_image_generator\main.py input.jpg noisy-img.jpg --noise_std 25
```

### 2. Create environment variable for denoising kernel

Create an environment variable called `DENOISING_KERNEL_PATH` and set its value to the absolute path of the `.\TotalVariationDenoising\GPU_Denoising\Denoising.cl` file. This ensures that you can run the project from anywhere.

### 3. Build the denoising executable from the C++ file.

The simplest way to build the project is using Visual Studio.

### 4. Run the Denoising Executable

After building, you can run the denoising executable from the command line. Example usage:

```sh
.\TotalVariationDenoising\x64\Release\GPU_Denoising.exe input.jpg output.jpg 0.1 0.01 0.0032 false
```
- The arguments are:  
  `input_image_path output_image_path strength step_size tolerance suppress_log`

### 5. Use the Python GUI

Use the GUI to select the input image, set the output path, adjust parameters, and select the denoising executable. You can launch the GUI to interactively select images and parameters:

```sh
py .\denoise_gui\main.py
```
