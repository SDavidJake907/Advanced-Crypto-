#include "cuda_features.hpp"

#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>

__global__ void momentum_kernel(
    const double* __restrict__ prices,
    double* __restrict__ out_mom,
    int n_assets,
    int n_points,
    int lookback_mom
) {
    const int asset = blockIdx.x * blockDim.x + threadIdx.x;
    if (asset >= n_assets) {
        return;
    }

    const int idx_last = asset * n_points + (n_points - 1);
    const int idx_lb = asset * n_points + (n_points - 1 - lookback_mom);
    const double p_last = prices[idx_last];
    const double p_lb = prices[idx_lb];

    out_mom[asset] = (p_lb > 0.0) ? ((p_last / p_lb) - 1.0) : 0.0;
}

__global__ void volatility_kernel(
    const double* __restrict__ prices,
    double* __restrict__ out_vol,
    int n_assets,
    int n_points,
    int lookback_vol
) {
    const int asset = blockIdx.x * blockDim.x + threadIdx.x;
    if (asset >= n_assets) {
        return;
    }

    const int start = n_points - lookback_vol;
    double mean = 0.0;
    double m2 = 0.0;
    int count = 0;

    for (int i = start; i < n_points; ++i) {
        const double x = prices[asset * n_points + i];
        ++count;
        const double delta = x - mean;
        mean += delta / count;
        const double delta2 = x - mean;
        m2 += delta * delta2;
    }

    const double var = (count > 1) ? (m2 / (count - 1)) : 0.0;
    out_vol[asset] = sqrt(var);
}

static void check_cuda(cudaError_t err) {
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

FeatureOutput compute_features_gpu(
    const std::vector<double>& prices,
    std::size_t n_assets,
    std::size_t n_points,
    const FeatureConfig& cfg
) {
    FeatureOutput out;
    out.momentum.resize(n_assets);
    out.volatility.resize(n_assets);

    const std::size_t total = n_assets * n_points;
    const std::size_t bytes = total * sizeof(double);
    const std::size_t bytes_vec = n_assets * sizeof(double);

    double* d_prices = nullptr;
    double* d_mom = nullptr;
    double* d_vol = nullptr;

    check_cuda(cudaMalloc(&d_prices, bytes));
    check_cuda(cudaMalloc(&d_mom, bytes_vec));
    check_cuda(cudaMalloc(&d_vol, bytes_vec));

    try {
        check_cuda(cudaMemcpy(d_prices, prices.data(), bytes, cudaMemcpyHostToDevice));

        const int threads = 128;
        const int blocks = static_cast<int>((n_assets + threads - 1) / threads);

        momentum_kernel<<<blocks, threads>>>(
            d_prices,
            d_mom,
            static_cast<int>(n_assets),
            static_cast<int>(n_points),
            cfg.lookback_mom
        );
        volatility_kernel<<<blocks, threads>>>(
            d_prices,
            d_vol,
            static_cast<int>(n_assets),
            static_cast<int>(n_points),
            cfg.lookback_vol
        );

        check_cuda(cudaGetLastError());
        check_cuda(cudaDeviceSynchronize());

        check_cuda(cudaMemcpy(out.momentum.data(), d_mom, bytes_vec, cudaMemcpyDeviceToHost));
        check_cuda(cudaMemcpy(out.volatility.data(), d_vol, bytes_vec, cudaMemcpyDeviceToHost));
    } catch (...) {
        cudaFree(d_prices);
        cudaFree(d_mom);
        cudaFree(d_vol);
        throw;
    }

    cudaFree(d_prices);
    cudaFree(d_mom);
    cudaFree(d_vol);

    return out;
}

