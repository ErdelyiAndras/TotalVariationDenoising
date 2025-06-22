__kernel void sum_int(__global int* data, int offset)
{
    int idx = get_global_id(0);
    int right_idx = idx + offset;
    data[idx] += data[right_idx];
}

__kernel void sum_float(__global float* data, int offset)
{
    int idx = get_global_id(0);
    int right_idx = idx + offset;
    data[idx] += data[right_idx];
}

__kernel void tv_norm_mtx_and_dx_dy(
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

    tv_norm_mtx[idx] = grad_mag;

    const float dx = x_diff / grad_mag;
    const float dy = y_diff / grad_mag;

    dx_mtx[idx] = dx;
    dy_mtx[idx] = dy;
}

__kernel void grad_from_dx_dy_step1(
    __global const float* dx, 
    __global const float* dy, 
    __global float* grad,
    int rows,
    int cols
) {
    const int idx = get_global_id(0);
    const int i = idx / cols;
    const int j = idx % cols;

    if (idx >= rows * cols) {
        return;
    }
    grad[idx] = 0.0f;

    if (i >= rows - 1 || j >= cols - 1) {
        return;
    }

    grad[idx] += dx[idx] + dy[idx];
}

__kernel void grad_from_dx_dy_step2(
    __global const float* dx,
    __global const float* dy,
    __global float* grad,
    int rows,
    int cols
) {
    const int idx = get_global_id(0);
    const int i = idx / cols;
    const int j = idx % cols;

    if (i >= rows - 1 || j >= cols - 1) {
        return;
    }

    const int idx_right = idx + 1;

    grad[idx_right] -= dx[idx];
}

__kernel void grad_from_dx_dy_step3(
    __global const float* dx,
    __global const float* dy,
    __global float* grad,
    int rows,
    int cols
) {
    const int idx = get_global_id(0);
    const int i = idx / cols;
    const int j = idx % cols;

    if (i >= rows - 1 || j >= cols - 1) {
        return;
    }

    const int idx_down = idx + cols;

    grad[idx_down] -= dy[idx];
}

__kernel void l2_norm_mtx_and_grad(
    __global const float* img,
    __global const float* orig,
    __global float* l2_norm_mtx,
    __global float* grad,
    int rows,
    int cols
) {
    int idx = get_global_id(0);
    if (idx >= rows * cols) {
        return;
    }
    float diff = img[idx] - orig[idx];
    grad[idx] = diff;
    l2_norm_mtx[idx] = diff * diff;
}

__kernel void eval_loss_and_grad(
    __global float* grad,
    __global const float* tv_or_l2_grad,
    float strength
) {
    int idx = get_global_id(0);
    grad[idx] += strength * tv_or_l2_grad[idx];
}

__kernel void eval_momentum(
    __global float* momentum,
    __global const float* grad,
    float momentum_beta
) {
    int idx = get_global_id(0);
    momentum[idx] *= momentum_beta;
    momentum[idx] += grad[idx] * (1.0f - momentum_beta);
}

__kernel void update_img(
    __global float* img,
    __global const float* momentum,
    float step,
    float momentum_beta,
    int counter
) {
    int idx = get_global_id(0);
    float bias_correction = 1.0f - pow(momentum_beta, (float)counter);
    img[idx] -= step / bias_correction * momentum[idx];
}