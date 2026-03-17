#include "cuda_northstar.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kNBins = 10;
constexpr int kMaxFingerprintCoins = 128;
constexpr int kFingerprintMetrics = 8;
constexpr int kHurstSizes[] = {8, 16, 32, 64, 128};
constexpr int kMaxHurstSizes = sizeof(kHurstSizes) / sizeof(kHurstSizes[0]);

void check_cuda(cudaError_t err) {
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

__global__ void compute_returns(
    const double* __restrict__ prices,
    double* __restrict__ returns,
    int n_coins,
    int n_samples
) {
    const int coin = blockIdx.x;
    const int n_ret = n_samples - 1;
    if (coin >= n_coins) {
        return;
    }

    for (int j = threadIdx.x; j < n_ret; j += blockDim.x) {
        const int idx = coin * n_samples + j;
        const double p = prices[idx];
        const double p_next = prices[idx + 1];
        const double denom = fmax(p, 1e-9);
        returns[coin * n_ret + j] = (p_next - p) / denom;
    }
}

__global__ void compute_log_returns(
    const double* __restrict__ prices,
    double* __restrict__ log_returns,
    int n_coins,
    int n_samples
) {
    const int coin = blockIdx.x;
    const int n_ret = n_samples - 1;
    if (coin >= n_coins) {
        return;
    }

    for (int j = threadIdx.x; j < n_ret; j += blockDim.x) {
        const int idx = coin * n_samples + j;
        const double lp0 = log(fmax(prices[idx], 1e-15));
        const double lp1 = log(fmax(prices[idx + 1], 1e-15));
        log_returns[coin * n_ret + j] = lp1 - lp0;
    }
}

__global__ void hurst_rs_blocks(
    const double* __restrict__ log_returns,
    double* __restrict__ rs_out,
    int n_returns,
    int block_size,
    int n_blocks,
    int size_idx,
    int max_sizes,
    int n_coins
) {
    const int flat_id = blockIdx.x;
    const int coin = flat_id / n_blocks;
    const int blk = flat_id % n_blocks;
    if (coin >= n_coins || blk >= n_blocks) {
        return;
    }

    extern __shared__ double sdata[];
    double* block_data = sdata;
    int tid = threadIdx.x;
    if (tid >= block_size) {
        return;
    }

    const int offset = coin * n_returns + blk * block_size + tid;
    const double val = (offset < coin * n_returns + n_returns) ? log_returns[offset] : 0.0;
    block_data[tid] = val;
    __syncthreads();

    __shared__ double s_mean;
    __shared__ double s_std;
    __shared__ double s_range;
    if (tid == 0) {
        double sum = 0.0;
        for (int i = 0; i < block_size; ++i) {
            sum += block_data[i];
        }
        s_mean = sum / block_size;

        double var_sum = 0.0;
        for (int i = 0; i < block_size; ++i) {
            const double d = block_data[i] - s_mean;
            var_sum += d * d;
        }
        s_std = sqrt(var_sum / block_size);

        double cum = 0.0;
        double mn = 1e30;
        double mx = -1e30;
        for (int i = 0; i < block_size; ++i) {
            cum += block_data[i] - s_mean;
            if (cum < mn) {
                mn = cum;
            }
            if (cum > mx) {
                mx = cum;
            }
        }
        s_range = mx - mn;
    }
    __syncthreads();

    if (tid == 0 && s_std > 1e-15) {
        const double rs = s_range / s_std;
        atomicAdd(&rs_out[coin * max_sizes + size_idx], rs / static_cast<double>(n_blocks));
    }
}

__global__ void entropy_minmax(
    const double* __restrict__ returns,
    double* __restrict__ minmax_out,
    int n_coins,
    int n_returns
) {
    const int coin = blockIdx.x;
    if (coin >= n_coins) {
        return;
    }

    __shared__ double s_min[256];
    __shared__ double s_max[256];
    const int tid = threadIdx.x;

    double local_min = 1e30;
    double local_max = -1e30;
    for (int j = tid; j < n_returns; j += blockDim.x) {
        const double v = returns[coin * n_returns + j];
        if (v < local_min) {
            local_min = v;
        }
        if (v > local_max) {
            local_max = v;
        }
    }
    s_min[tid] = local_min;
    s_max[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_min[tid + s] < s_min[tid]) {
                s_min[tid] = s_min[tid + s];
            }
            if (s_max[tid + s] > s_max[tid]) {
                s_max[tid] = s_max[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        minmax_out[coin * 2] = s_min[0];
        minmax_out[coin * 2 + 1] = s_max[0];
    }
}

__global__ void entropy_histogram(
    const double* __restrict__ returns,
    const double* __restrict__ minmax,
    int* __restrict__ bins_out,
    int n_coins,
    int n_returns
) {
    const int coin = blockIdx.x;
    if (coin >= n_coins) {
        return;
    }

    const double r_min = minmax[coin * 2];
    const double r_max = minmax[coin * 2 + 1];
    const double rng = r_max - r_min;

    __shared__ int s_bins[kNBins];
    const int tid = threadIdx.x;
    if (tid < kNBins) {
        s_bins[tid] = 0;
    }
    __syncthreads();

    if (rng < 1e-15) {
        if (tid == 0) {
            atomicAdd(&bins_out[coin * kNBins], n_returns);
        }
        return;
    }

    for (int j = tid; j < n_returns; j += blockDim.x) {
        const double v = returns[coin * n_returns + j];
        int bin = static_cast<int>(floor((v - r_min) / rng * (kNBins - 1)));
        if (bin < 0) {
            bin = 0;
        }
        if (bin >= kNBins) {
            bin = kNBins - 1;
        }
        atomicAdd(&s_bins[bin], 1);
    }
    __syncthreads();

    if (tid < kNBins) {
        atomicAdd(&bins_out[coin * kNBins + tid], s_bins[tid]);
    }
}

__global__ void autocorr_stats(
    const double* __restrict__ returns,
    double* __restrict__ stats_out,
    int n_coins,
    int n_returns
) {
    const int coin = blockIdx.x;
    if (coin >= n_coins) {
        return;
    }

    __shared__ double s_sum[256];
    __shared__ double s_var[256];
    __shared__ double s_cov[256];
    const int tid = threadIdx.x;

    double local_sum = 0.0;
    for (int j = tid; j < n_returns; j += blockDim.x) {
        local_sum += returns[coin * n_returns + j];
    }
    s_sum[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    __shared__ double s_mean;
    if (tid == 0) {
        s_mean = s_sum[0] / n_returns;
    }
    __syncthreads();

    double local_var = 0.0;
    double local_cov = 0.0;
    for (int j = tid; j < n_returns; j += blockDim.x) {
        const double d = returns[coin * n_returns + j] - s_mean;
        local_var += d * d;
        if (j > 0) {
            const double d_prev = returns[coin * n_returns + j - 1] - s_mean;
            local_cov += d * d_prev;
        }
    }
    s_var[tid] = local_var;
    s_cov[tid] = local_cov;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_var[tid] += s_var[tid + s];
            s_cov[tid] += s_cov[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        stats_out[coin * 3 + 0] = s_mean;
        stats_out[coin * 3 + 1] = s_var[0] / n_returns;
        stats_out[coin * 3 + 2] = s_cov[0] / (n_returns - 1);
    }
}

__global__ void fingerprint_latest(
    const double* __restrict__ returns,
    double* __restrict__ metrics_out,
    int n_coins,
    int n_returns,
    int btc_idx,
    int eth_idx
) {
    __shared__ double latest[kMaxFingerprintCoins];
    __shared__ double sorted[kMaxFingerprintCoins];
    __shared__ double s_rv[256];
    __shared__ double s_scalar[6];
    __shared__ int s_rv_n;

    const int tid = threadIdx.x;
    if (tid < n_coins && tid < kMaxFingerprintCoins) {
        latest[tid] = returns[tid * n_returns + (n_returns - 1)];
        sorted[tid] = latest[tid];
    }
    __syncthreads();

    if (tid == 0) {
        double sum = 0.0;
        int pos_count = 0;
        for (int i = 0; i < n_coins; ++i) {
            sum += latest[i];
            if (latest[i] > 0) {
                pos_count++;
            }
        }
        s_scalar[0] = sum / n_coins;
        s_scalar[1] = (btc_idx < n_coins) ? latest[btc_idx] : 0.0;
        s_scalar[2] = (eth_idx < n_coins) ? latest[eth_idx] : 0.0;
        s_scalar[3] = static_cast<double>(pos_count) / n_coins;

        for (int i = 1; i < n_coins; ++i) {
            const double key = sorted[i];
            int j = i - 1;
            while (j >= 0 && sorted[j] > key) {
                sorted[j + 1] = sorted[j];
                --j;
            }
            sorted[j + 1] = key;
        }
        const int n = n_coins;
        const double median = (n % 2) ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
        const double iqr = (n >= 4) ? (sorted[3 * n / 4] - sorted[n / 4]) : 0.0;
        s_scalar[4] = median;
        s_scalar[5] = iqr;
        s_rv_n = (n_returns < 512) ? n_returns : 512;
    }
    __syncthreads();

    const int rv_n = s_rv_n;
    double local_sum = 0.0;
    for (int t = tid; t < rv_n; t += blockDim.x) {
        double mkt = 0.0;
        for (int c = 0; c < n_coins; ++c) {
            mkt += returns[c * n_returns + t];
        }
        mkt /= n_coins;
        local_sum += mkt * mkt;
    }
    s_rv[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_rv[tid] += s_rv[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        metrics_out[0] = s_scalar[0];
        metrics_out[1] = s_scalar[1];
        metrics_out[2] = s_scalar[2];
        metrics_out[3] = s_scalar[3];
        metrics_out[4] = s_scalar[4];
        metrics_out[5] = s_scalar[5];
        metrics_out[6] = (rv_n > 0) ? sqrt(s_rv[0] / rv_n) : 0.0;

        double corr_sum = 0.0;
        int corr_count = 0;
        if (n_coins > 2 && n_returns >= 10) {
            const double* btc_ret = &returns[btc_idx * n_returns];
            double btc_mean = 0.0;
            for (int t = 0; t < n_returns; ++t) {
                btc_mean += btc_ret[t];
            }
            btc_mean /= n_returns;
            double btc_var = 0.0;
            for (int t = 0; t < n_returns; ++t) {
                const double d = btc_ret[t] - btc_mean;
                btc_var += d * d;
            }
            const double btc_std = sqrt(btc_var / n_returns);

            for (int c = 0; c < n_coins; ++c) {
                if (c == btc_idx || c == eth_idx) {
                    continue;
                }
                const double* alt_ret = &returns[c * n_returns];
                double alt_mean = 0.0;
                for (int t = 0; t < n_returns; ++t) {
                    alt_mean += alt_ret[t];
                }
                alt_mean /= n_returns;
                double alt_var = 0.0;
                double cov = 0.0;
                for (int t = 0; t < n_returns; ++t) {
                    const double db = btc_ret[t] - btc_mean;
                    const double da = alt_ret[t] - alt_mean;
                    alt_var += da * da;
                    cov += db * da;
                }
                const double alt_std = sqrt(alt_var / n_returns);
                const double denom = btc_std * alt_std * n_returns;
                if (denom > 1e-15) {
                    const double r = cov / denom;
                    if (r == r) {
                        corr_sum += r;
                        corr_count++;
                    }
                }
            }
        }
        metrics_out[7] = (corr_count > 0) ? corr_sum / corr_count : 0.0;
    }
}

std::vector<double> fit_hurst_from_rs(const std::vector<double>& rs_host, std::size_t n_assets) {
    std::vector<double> hurst(n_assets, 0.5);
    for (std::size_t asset = 0; asset < n_assets; ++asset) {
        std::vector<double> log_ns;
        std::vector<double> log_rs;
        for (int size_idx = 0; size_idx < kMaxHurstSizes; ++size_idx) {
            const double rs = rs_host[asset * kMaxHurstSizes + size_idx];
            if (rs > 0.0) {
                log_ns.push_back(std::log(static_cast<double>(kHurstSizes[size_idx])));
                log_rs.push_back(std::log(rs));
            }
        }
        if (log_ns.size() < 2) {
            continue;
        }

        const double n = static_cast<double>(log_ns.size());
        double sx = 0.0;
        double sy = 0.0;
        double sxy = 0.0;
        double sxx = 0.0;
        for (std::size_t i = 0; i < log_ns.size(); ++i) {
            sx += log_ns[i];
            sy += log_rs[i];
            sxy += log_ns[i] * log_rs[i];
            sxx += log_ns[i] * log_ns[i];
        }
        const double denom = (n * sxx - sx * sx);
        if (std::abs(denom) < 1e-15) {
            continue;
        }
        const double slope = (n * sxy - sx * sy) / denom;
        hurst[asset] = std::max(0.0, std::min(1.0, slope));
    }
    return hurst;
}

std::vector<double> entropy_from_bins(const std::vector<int>& bins_host, std::size_t n_assets) {
    std::vector<double> entropies(n_assets, 0.5);
    const double max_entropy = std::log2(static_cast<double>(kNBins));
    for (std::size_t asset = 0; asset < n_assets; ++asset) {
        int total = 0;
        for (int b = 0; b < kNBins; ++b) {
            total += bins_host[asset * kNBins + b];
        }
        if (total == 0) {
            continue;
        }

        double entropy = 0.0;
        for (int b = 0; b < kNBins; ++b) {
            const int count = bins_host[asset * kNBins + b];
            if (count <= 0) {
                continue;
            }
            const double p = static_cast<double>(count) / static_cast<double>(total);
            entropy -= p * std::log2(p);
        }
        entropies[asset] = (max_entropy > 0.0) ? std::max(0.0, std::min(1.0, entropy / max_entropy)) : 0.5;
    }
    return entropies;
}

std::vector<double> autocorr_from_stats(const std::vector<double>& stats_host, std::size_t n_assets) {
    std::vector<double> autocorr(n_assets, 0.0);
    for (std::size_t asset = 0; asset < n_assets; ++asset) {
        const double variance = stats_host[asset * 3 + 1];
        const double cov = stats_host[asset * 3 + 2];
        if (variance > 1e-15) {
            double value = cov / variance;
            if (value > 1.0) {
                value = 1.0;
            }
            if (value < -1.0) {
                value = -1.0;
            }
            autocorr[asset] = value;
        }
    }
    return autocorr;
}

}  // namespace

NorthstarBatchFeatureOutput compute_northstar_batch_features_gpu(
    const std::vector<double>& prices,
    std::size_t n_assets,
    std::size_t n_points
) {
    NorthstarBatchFeatureOutput out;
    out.hurst.assign(n_assets, 0.5);
    out.entropy.assign(n_assets, 0.5);
    out.autocorr.assign(n_assets, 0.0);

    if (n_assets == 0 || n_points < 3) {
        return out;
    }

    const std::size_t total = n_assets * n_points;
    const std::size_t n_returns = n_points - 1;
    const std::size_t price_bytes = total * sizeof(double);
    const std::size_t returns_bytes = n_assets * n_returns * sizeof(double);

    double* d_prices = nullptr;
    double* d_returns = nullptr;
    double* d_log_returns = nullptr;
    double* d_rs_out = nullptr;
    double* d_minmax = nullptr;
    int* d_bins = nullptr;
    double* d_ac_stats = nullptr;

    check_cuda(cudaMalloc(&d_prices, price_bytes));
    check_cuda(cudaMalloc(&d_returns, returns_bytes));
    check_cuda(cudaMalloc(&d_log_returns, returns_bytes));
    check_cuda(cudaMalloc(&d_rs_out, n_assets * kMaxHurstSizes * sizeof(double)));
    check_cuda(cudaMalloc(&d_minmax, n_assets * 2 * sizeof(double)));
    check_cuda(cudaMalloc(&d_bins, n_assets * kNBins * sizeof(int)));
    check_cuda(cudaMalloc(&d_ac_stats, n_assets * 3 * sizeof(double)));

    try {
        check_cuda(cudaMemcpy(d_prices, prices.data(), price_bytes, cudaMemcpyHostToDevice));
        check_cuda(cudaMemset(d_rs_out, 0, n_assets * kMaxHurstSizes * sizeof(double)));
        check_cuda(cudaMemset(d_bins, 0, n_assets * kNBins * sizeof(int)));

        const int threads = static_cast<int>(std::min<std::size_t>(256, n_returns));
        compute_returns<<<static_cast<unsigned int>(n_assets), static_cast<unsigned int>(threads)>>>(
            d_prices,
            d_returns,
            static_cast<int>(n_assets),
            static_cast<int>(n_points)
        );
        compute_log_returns<<<static_cast<unsigned int>(n_assets), static_cast<unsigned int>(threads)>>>(
            d_prices,
            d_log_returns,
            static_cast<int>(n_assets),
            static_cast<int>(n_points)
        );
        check_cuda(cudaGetLastError());

        for (int size_idx = 0; size_idx < kMaxHurstSizes; ++size_idx) {
            const int block_size = kHurstSizes[size_idx];
            if (block_size > static_cast<int>(n_returns)) {
                break;
            }
            const int n_blocks = static_cast<int>(n_returns) / block_size;
            if (n_blocks <= 0) {
                continue;
            }
            const int grid = static_cast<int>(n_assets) * n_blocks;
            const unsigned int shared_bytes = static_cast<unsigned int>(block_size * sizeof(double));
            hurst_rs_blocks<<<grid, block_size, shared_bytes>>>(
                d_log_returns,
                d_rs_out,
                static_cast<int>(n_returns),
                block_size,
                n_blocks,
                size_idx,
                kMaxHurstSizes,
                static_cast<int>(n_assets)
            );
            check_cuda(cudaGetLastError());
        }

        entropy_minmax<<<static_cast<unsigned int>(n_assets), 256>>>(
            d_returns,
            d_minmax,
            static_cast<int>(n_assets),
            static_cast<int>(n_returns)
        );
        entropy_histogram<<<static_cast<unsigned int>(n_assets), 256>>>(
            d_returns,
            d_minmax,
            d_bins,
            static_cast<int>(n_assets),
            static_cast<int>(n_returns)
        );
        autocorr_stats<<<static_cast<unsigned int>(n_assets), 256>>>(
            d_returns,
            d_ac_stats,
            static_cast<int>(n_assets),
            static_cast<int>(n_returns)
        );
        check_cuda(cudaGetLastError());
        check_cuda(cudaDeviceSynchronize());

        std::vector<double> rs_host(n_assets * kMaxHurstSizes, 0.0);
        std::vector<int> bins_host(n_assets * kNBins, 0);
        std::vector<double> ac_host(n_assets * 3, 0.0);
        check_cuda(cudaMemcpy(rs_host.data(), d_rs_out, rs_host.size() * sizeof(double), cudaMemcpyDeviceToHost));
        check_cuda(cudaMemcpy(bins_host.data(), d_bins, bins_host.size() * sizeof(int), cudaMemcpyDeviceToHost));
        check_cuda(cudaMemcpy(ac_host.data(), d_ac_stats, ac_host.size() * sizeof(double), cudaMemcpyDeviceToHost));

        out.hurst = fit_hurst_from_rs(rs_host, n_assets);
        out.entropy = entropy_from_bins(bins_host, n_assets);
        out.autocorr = autocorr_from_stats(ac_host, n_assets);
    } catch (...) {
        cudaFree(d_prices);
        cudaFree(d_returns);
        cudaFree(d_log_returns);
        cudaFree(d_rs_out);
        cudaFree(d_minmax);
        cudaFree(d_bins);
        cudaFree(d_ac_stats);
        throw;
    }

    cudaFree(d_prices);
    cudaFree(d_returns);
    cudaFree(d_log_returns);
    cudaFree(d_rs_out);
    cudaFree(d_minmax);
    cudaFree(d_bins);
    cudaFree(d_ac_stats);
    return out;
}

NorthstarFingerprintOutput compute_northstar_fingerprint_gpu(
    const std::vector<double>& prices,
    std::size_t n_assets,
    std::size_t n_points,
    int btc_idx,
    int eth_idx
) {
    NorthstarFingerprintOutput out;
    out.metrics.assign(kFingerprintMetrics, 0.0);

    if (n_assets == 0 || n_points < 3 || n_assets > kMaxFingerprintCoins) {
        return out;
    }

    const std::size_t total = n_assets * n_points;
    const std::size_t n_returns = n_points - 1;
    const std::size_t price_bytes = total * sizeof(double);
    const std::size_t returns_bytes = n_assets * n_returns * sizeof(double);

    double* d_prices = nullptr;
    double* d_returns = nullptr;
    double* d_metrics = nullptr;
    check_cuda(cudaMalloc(&d_prices, price_bytes));
    check_cuda(cudaMalloc(&d_returns, returns_bytes));
    check_cuda(cudaMalloc(&d_metrics, kFingerprintMetrics * sizeof(double)));

    try {
        check_cuda(cudaMemcpy(d_prices, prices.data(), price_bytes, cudaMemcpyHostToDevice));
        compute_returns<<<static_cast<unsigned int>(n_assets), static_cast<unsigned int>(std::min<std::size_t>(256, n_returns))>>>(
            d_prices,
            d_returns,
            static_cast<int>(n_assets),
            static_cast<int>(n_points)
        );
        fingerprint_latest<<<1, 256>>>(
            d_returns,
            d_metrics,
            static_cast<int>(n_assets),
            static_cast<int>(n_returns),
            btc_idx,
            eth_idx
        );
        check_cuda(cudaGetLastError());
        check_cuda(cudaDeviceSynchronize());
        check_cuda(cudaMemcpy(out.metrics.data(), d_metrics, kFingerprintMetrics * sizeof(double), cudaMemcpyDeviceToHost));
    } catch (...) {
        cudaFree(d_prices);
        cudaFree(d_returns);
        cudaFree(d_metrics);
        throw;
    }

    cudaFree(d_prices);
    cudaFree(d_returns);
    cudaFree(d_metrics);
    return out;
}
