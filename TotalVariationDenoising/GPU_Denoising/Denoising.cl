__kernel void sum_int(__global int* data, int offset, int size)
{
    int idx = get_global_id(0);

    __global int* left = data + idx;
    if (idx + offset >= size) {
        return;
    }
    __global int* right = left + offset;
    *left += *right;
}

__kernel void sum_float(__global float* data, int offset, int size)
{
    int idx = get_global_id(0);

    __global float* left = data + idx;
    if (idx + offset >= size) {
        return;
    }
    __global float* right = left + offset;
    *left += *right;
}

__kernel void tv_norm_mtx_and_dx_dy
(
    __global const float* img,
    __global float* tv_norm_mtx,
    __global float* dx_mtx,
    __global float* dy_mtx,
    int rows,
    int cols,
    float eps
) {
    const int idx = get_global_id(0);
    const int i = idx / cols;
    const int j = idx % cols;

    if (i >= rows - 1 || j >= cols - 1) {
        return;
    }

    const int idx_right = idx + 1;
    const int idx_down = idx + cols;

    const float x_diff = img[idx] - img[idx_right];
    const float y_diff = img[idx] - img[idx_down];
    const float grad_mag = sqrt(x_diff * x_diff + y_diff * y_diff + eps);

    tv_norm_mtx[idx] = grad_mag; // need to sum up 

    const float dx = x_diff / grad_mag;
    const float dy = y_diff / grad_mag;

    dx_mtx[idx] = dx;
    dy_mtx[idx] = dy;
}

__kernel void l2_norm_and_grad
(
    __global const float* img,
    __global const float* orig,
    __global float* grad,
    __global float* l2_norm,
    int rows,
    int cols
) {
    int idx = get_global_id(0);
    if (idx >= rows * cols) return;
    float diff = img[idx] - orig[idx];
    grad[idx] = diff;
    //atomic_fetch_add(l2_norm, diff * diff);
}

