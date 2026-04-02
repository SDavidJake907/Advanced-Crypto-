#include "cuda_rsi.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

__global__ void rsi_kernel(
    const double* __restrict__ prices,
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

    // Wilder's RSI: seed with SMA of first `lookback` deltas
    double avg_gain = 0.0;
    double avg_loss = 0.0;
    for (int i = 1; i <= lookback; ++i) {
        const double delta = prices[base + i] - prices[base + i - 1];
        if (delta > 0.0) {
            avg_gain += delta;
        } else {
            avg_loss -= delta;
        }
    }
    avg_gain /= lookback;
    avg_loss /= lookback;

    // Apply Wilder's RMA smoothing (alpha = 1/lookback) for remaining bars
    for (int i = lookback + 1; i < n_points; ++i) {
        const double delta = prices[base + i] - prices[base + i - 1];
        const double gain = (delta > 0.0) ? delta : 0.0;
        const double loss = (delta < 0.0) ? -delta : 0.0;
        avg_gain = (avg_gain * (lookback - 1) + gain) / lookback;
        avg_loss = (avg_loss * (lookback - 1) + loss) / lookback;
    }

    if (avg_loss == 0.0) {
        out[asset] = 100.0;
        return;
    }
    const double rs = avg_gain / avg_loss;
    out[asset] = 100.0 - (100.0 / (1.0 + rs));
}

static void check_cuda(cudaError_t err) {
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

std::vector<double> compute_rsi_gpu(
    const std::vector<double>& prices,
    std::size_t n_assets,
    std::size_t n_points,
    int lookback
) {
    std::vector<double> out(n_assets, 0.0);
    const std::size_t in_bytes = n_assets * n_points * sizeof(double);
    const std::size_t out_bytes = n_assets * sizeof(double);
    double* d_prices = nullptr;
    double* d_out = nullptr;
    check_cuda(cudaMalloc(&d_prices, in_bytes));
    check_cuda(cudaMalloc(&d_out, out_bytes));
    try {
        check_cuda(cudaMemcpy(d_prices, prices.data(), in_bytes, cudaMemcpyHostToDevice));
        const int threads = 128;
        const int blocks = static_cast<int>((n_assets + threads - 1) / threads);
        rsi_kernel<<<blocks, threads>>>(d_prices, d_out, static_cast<int>(n_assets), static_cast<int>(n_points), lookback);
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
