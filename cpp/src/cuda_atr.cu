#include "cuda_atr.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

__global__ void atr_kernel(
    const double* __restrict__ highs,
    const double* __restrict__ lows,
    const double* __restrict__ closes,
    double* __restrict__ out,
    int n_assets,
    int n_points,
    int lookback
) {
    const int asset = blockIdx.x * blockDim.x + threadIdx.x;
    if (asset >= n_assets) {
        return;
    }
    if (n_points <= lookback) {
        out[asset] = 0.0;
        return;
    }

    const int base = asset * n_points;
    double total = 0.0;
    for (int i = n_points - lookback; i < n_points; ++i) {
        const double high = highs[base + i];
        const double low = lows[base + i];
        const double prev_close = closes[base + i - 1];
        const double hl = high - low;
        const double hc = fabs(high - prev_close);
        const double lc = fabs(low - prev_close);
        total += fmax(hl, fmax(hc, lc));
    }
    out[asset] = total / lookback;
}

static void check_cuda(cudaError_t err) {
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

std::vector<double> compute_atr_gpu(
    const std::vector<double>& highs,
    const std::vector<double>& lows,
    const std::vector<double>& closes,
    std::size_t n_assets,
    std::size_t n_points,
    int lookback
) {
    std::vector<double> out(n_assets, 0.0);
    const std::size_t in_bytes = n_assets * n_points * sizeof(double);
    const std::size_t out_bytes = n_assets * sizeof(double);
    double* d_highs = nullptr;
    double* d_lows = nullptr;
    double* d_closes = nullptr;
    double* d_out = nullptr;
    check_cuda(cudaMalloc(&d_highs, in_bytes));
    check_cuda(cudaMalloc(&d_lows, in_bytes));
    check_cuda(cudaMalloc(&d_closes, in_bytes));
    check_cuda(cudaMalloc(&d_out, out_bytes));
    try {
        check_cuda(cudaMemcpy(d_highs, highs.data(), in_bytes, cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(d_lows, lows.data(), in_bytes, cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(d_closes, closes.data(), in_bytes, cudaMemcpyHostToDevice));
        const int threads = 128;
        const int blocks = static_cast<int>((n_assets + threads - 1) / threads);
        atr_kernel<<<blocks, threads>>>(d_highs, d_lows, d_closes, d_out, static_cast<int>(n_assets), static_cast<int>(n_points), lookback);
        check_cuda(cudaGetLastError());
        check_cuda(cudaDeviceSynchronize());
        check_cuda(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
    } catch (...) {
        cudaFree(d_highs);
        cudaFree(d_lows);
        cudaFree(d_closes);
        cudaFree(d_out);
        throw;
    }
    cudaFree(d_highs);
    cudaFree(d_lows);
    cudaFree(d_closes);
    cudaFree(d_out);
    return out;
}
