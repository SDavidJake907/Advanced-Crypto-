#include "cuda_correlation.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

__global__ void correlation_kernel(
    const double* __restrict__ prices,
    double* __restrict__ out_corr,
    int n_assets,
    int n_points
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_assets || col >= n_assets) {
        return;
    }

    double mean_x = 0.0;
    double mean_y = 0.0;
    for (int i = 0; i < n_points; ++i) {
        mean_x += prices[row * n_points + i];
        mean_y += prices[col * n_points + i];
    }
    mean_x /= n_points;
    mean_y /= n_points;

    double cov = 0.0;
    double var_x = 0.0;
    double var_y = 0.0;
    for (int i = 0; i < n_points; ++i) {
        const double dx = prices[row * n_points + i] - mean_x;
        const double dy = prices[col * n_points + i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    const double denom = sqrt(var_x * var_y);
    out_corr[row * n_assets + col] = denom > 0.0 ? (cov / denom) : 0.0;
}

static void check_cuda(cudaError_t err) {
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

std::vector<double> compute_correlation_gpu(
    const std::vector<double>& prices,
    std::size_t n_assets,
    std::size_t n_points
) {
    const std::size_t in_bytes = n_assets * n_points * sizeof(double);
    const std::size_t out_elems = n_assets * n_assets;
    const std::size_t out_bytes = out_elems * sizeof(double);

    std::vector<double> out(out_elems, 0.0);
    double* d_prices = nullptr;
    double* d_out = nullptr;

    check_cuda(cudaMalloc(&d_prices, in_bytes));
    check_cuda(cudaMalloc(&d_out, out_bytes));

    try {
        check_cuda(cudaMemcpy(d_prices, prices.data(), in_bytes, cudaMemcpyHostToDevice));
        const dim3 threads(16, 16);
        const dim3 blocks(
            static_cast<unsigned int>((n_assets + threads.x - 1) / threads.x),
            static_cast<unsigned int>((n_assets + threads.y - 1) / threads.y)
        );
        correlation_kernel<<<blocks, threads>>>(
            d_prices,
            d_out,
            static_cast<int>(n_assets),
            static_cast<int>(n_points)
        );
        check_cuda(cudaGetLastError());
        check_cuda(cudaDeviceSynchronize());
        check_cuda(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
    } catch (...) {
        cudaFree(d_prices);
        cudaFree(d_out);
        throw;
    }

    cudaFree(d_prices);
    cudaFree(d_out);
    return out;
}
