#include "cuda_bollinger.hpp"

#include <cuda_runtime.h>
#include <stdexcept>

__global__ void bollinger_kernel(
    const double* __restrict__ prices,
    double* __restrict__ middle,
    double* __restrict__ upper,
    double* __restrict__ lower,
    double* __restrict__ bandwidth,
    int n_assets,
    int n_points,
    int lookback,
    double num_std
) {
    const int asset = blockIdx.x * blockDim.x + threadIdx.x;
    if (asset >= n_assets) {
        return;
    }
    if (n_points < lookback) {
        middle[asset] = 0.0;
        upper[asset] = 0.0;
        lower[asset] = 0.0;
        bandwidth[asset] = 0.0;
        return;
    }

    const int base = asset * n_points;
    double mean = 0.0;
    for (int i = n_points - lookback; i < n_points; ++i) {
        mean += prices[base + i];
    }
    mean /= lookback;

    double var = 0.0;
    for (int i = n_points - lookback; i < n_points; ++i) {
        const double d = prices[base + i] - mean;
        var += d * d;
    }
    const double std = sqrt(var / lookback);
    const double up = mean + (num_std * std);
    const double lo = mean - (num_std * std);

    middle[asset] = mean;
    upper[asset] = up;
    lower[asset] = lo;
    bandwidth[asset] = mean != 0.0 ? (up - lo) / mean : 0.0;
}

static void check_cuda(cudaError_t err) {
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

BollingerOutput compute_bollinger_gpu(
    const std::vector<double>& prices,
    std::size_t n_assets,
    std::size_t n_points,
    int lookback,
    double num_std
) {
    BollingerOutput out;
    out.middle.resize(n_assets);
    out.upper.resize(n_assets);
    out.lower.resize(n_assets);
    out.bandwidth.resize(n_assets);

    const std::size_t in_bytes = n_assets * n_points * sizeof(double);
    const std::size_t out_bytes = n_assets * sizeof(double);
    double* d_prices = nullptr;
    double* d_middle = nullptr;
    double* d_upper = nullptr;
    double* d_lower = nullptr;
    double* d_bandwidth = nullptr;
    check_cuda(cudaMalloc(&d_prices, in_bytes));
    check_cuda(cudaMalloc(&d_middle, out_bytes));
    check_cuda(cudaMalloc(&d_upper, out_bytes));
    check_cuda(cudaMalloc(&d_lower, out_bytes));
    check_cuda(cudaMalloc(&d_bandwidth, out_bytes));
    try {
        check_cuda(cudaMemcpy(d_prices, prices.data(), in_bytes, cudaMemcpyHostToDevice));
        const int threads = 128;
        const int blocks = static_cast<int>((n_assets + threads - 1) / threads);
        bollinger_kernel<<<blocks, threads>>>(
            d_prices, d_middle, d_upper, d_lower, d_bandwidth,
            static_cast<int>(n_assets), static_cast<int>(n_points), lookback, num_std
        );
        check_cuda(cudaGetLastError());
        check_cuda(cudaDeviceSynchronize());
        check_cuda(cudaMemcpy(out.middle.data(), d_middle, out_bytes, cudaMemcpyDeviceToHost));
        check_cuda(cudaMemcpy(out.upper.data(), d_upper, out_bytes, cudaMemcpyDeviceToHost));
        check_cuda(cudaMemcpy(out.lower.data(), d_lower, out_bytes, cudaMemcpyDeviceToHost));
        check_cuda(cudaMemcpy(out.bandwidth.data(), d_bandwidth, out_bytes, cudaMemcpyDeviceToHost));
    } catch (...) {
        cudaFree(d_prices);
        cudaFree(d_middle);
        cudaFree(d_upper);
        cudaFree(d_lower);
        cudaFree(d_bandwidth);
        throw;
    }
    cudaFree(d_prices);
    cudaFree(d_middle);
    cudaFree(d_upper);
    cudaFree(d_lower);
    cudaFree(d_bandwidth);
    return out;
}
