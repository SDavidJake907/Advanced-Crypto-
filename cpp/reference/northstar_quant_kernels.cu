/**
 * quant_kernels.cu — GPU-accelerated quant math from Northstar.
 *
 * Reference import from:
 * C:\Users\kitti\Downloads\-Northstar-crypto-source--main (1).zip
 * path inside archive:
 * -Northstar-crypto-source--main/rust_core/cuda/quant_kernels.cu
 *
 * Data layout: row-major, coin i sample j at [i * n_samples + j].
 * All buffers are f64 (double precision).
 */

extern "C" {

__global__ void compute_returns(
    const double* __restrict__ prices,
    double* __restrict__ returns,
    int n_coins,
    int n_samples
) {
    int coin = blockIdx.x;
    int n_ret = n_samples - 1;
    if (coin >= n_coins) return;

    for (int j = threadIdx.x; j < n_ret; j += blockDim.x) {
        int idx = coin * n_samples + j;
        double p = prices[idx];
        double p_next = prices[idx + 1];
        double denom = fmax(p, 1e-9);
        returns[coin * n_ret + j] = (p_next - p) / denom;
    }
}

__global__ void compute_log_returns(
    const double* __restrict__ prices,
    double* __restrict__ log_returns,
    int n_coins,
    int n_samples
) {
    int coin = blockIdx.x;
    int n_ret = n_samples - 1;
    if (coin >= n_coins) return;

    for (int j = threadIdx.x; j < n_ret; j += blockDim.x) {
        int idx = coin * n_samples + j;
        double lp0 = log(fmax(prices[idx], 1e-15));
        double lp1 = log(fmax(prices[idx + 1], 1e-15));
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
    int flat_id = blockIdx.x;
    int coin = flat_id / n_blocks;
    int blk = flat_id % n_blocks;
    if (coin >= n_coins || blk >= n_blocks) return;

    extern __shared__ double sdata[];
    double* block_data = sdata;
    double* cum_dev = sdata + block_size;

    int tid = threadIdx.x;
    if (tid >= block_size) return;

    int offset = coin * n_returns + blk * block_size + tid;
    double val = (offset < coin * n_returns + n_returns) ? log_returns[offset] : 0.0;
    block_data[tid] = val;
    __syncthreads();

    __shared__ double s_mean, s_std, s_range;
    if (tid == 0) {
        double sum = 0.0;
        for (int i = 0; i < block_size; i++) sum += block_data[i];
        s_mean = sum / block_size;

        double var_sum = 0.0;
        for (int i = 0; i < block_size; i++) {
            double d = block_data[i] - s_mean;
            var_sum += d * d;
        }
        s_std = sqrt(var_sum / block_size);

        double cum = 0.0;
        double mn = 1e30, mx = -1e30;
        for (int i = 0; i < block_size; i++) {
            cum += block_data[i] - s_mean;
            if (cum < mn) mn = cum;
            if (cum > mx) mx = cum;
        }
        s_range = mx - mn;
    }
    __syncthreads();

    if (tid == 0 && s_std > 1e-15) {
        double rs = s_range / s_std;
        atomicAdd(&rs_out[coin * max_sizes + size_idx], rs / (double)n_blocks);
    }
}

#define N_BINS 10

__global__ void entropy_minmax(
    const double* __restrict__ returns,
    double* __restrict__ minmax_out,
    int n_coins,
    int n_returns
) {
    int coin = blockIdx.x;
    if (coin >= n_coins) return;

    __shared__ double s_min[256];
    __shared__ double s_max[256];
    int tid = threadIdx.x;

    double local_min = 1e30;
    double local_max = -1e30;

    for (int j = tid; j < n_returns; j += blockDim.x) {
        double v = returns[coin * n_returns + j];
        if (v < local_min) local_min = v;
        if (v > local_max) local_max = v;
    }
    s_min[tid] = local_min;
    s_max[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_min[tid + s] < s_min[tid]) s_min[tid] = s_min[tid + s];
            if (s_max[tid + s] > s_max[tid]) s_max[tid] = s_max[tid + s];
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
    int coin = blockIdx.x;
    if (coin >= n_coins) return;

    double r_min = minmax[coin * 2];
    double r_max = minmax[coin * 2 + 1];
    double rng = r_max - r_min;

    __shared__ int s_bins[N_BINS];
    int tid = threadIdx.x;
    if (tid < N_BINS) s_bins[tid] = 0;
    __syncthreads();

    if (rng < 1e-15) {
        if (tid == 0) atomicAdd(&bins_out[coin * N_BINS], n_returns);
        return;
    }

    for (int j = tid; j < n_returns; j += blockDim.x) {
        double v = returns[coin * n_returns + j];
        int bin = (int)floor((v - r_min) / rng * (N_BINS - 1));
        if (bin < 0) bin = 0;
        if (bin >= N_BINS) bin = N_BINS - 1;
        atomicAdd(&s_bins[bin], 1);
    }
    __syncthreads();

    if (tid < N_BINS) {
        atomicAdd(&bins_out[coin * N_BINS + tid], s_bins[tid]);
    }
}

__global__ void autocorr_stats(
    const double* __restrict__ returns,
    double* __restrict__ stats_out,
    int n_coins,
    int n_returns
) {
    int coin = blockIdx.x;
    if (coin >= n_coins) return;

    __shared__ double s_sum[256];
    __shared__ double s_var[256];
    __shared__ double s_cov[256];
    int tid = threadIdx.x;

    double local_sum = 0.0;
    for (int j = tid; j < n_returns; j += blockDim.x) {
        local_sum += returns[coin * n_returns + j];
    }
    s_sum[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }
    __shared__ double s_mean;
    if (tid == 0) s_mean = s_sum[0] / n_returns;
    __syncthreads();

    double local_var = 0.0;
    double local_cov = 0.0;
    for (int j = tid; j < n_returns; j += blockDim.x) {
        double d = returns[coin * n_returns + j] - s_mean;
        local_var += d * d;
        if (j > 0) {
            double d_prev = returns[coin * n_returns + j - 1] - s_mean;
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

__global__ void center_rows(
    const double* __restrict__ returns,
    double* __restrict__ centered,
    double* __restrict__ means_out,
    double* __restrict__ stds_out,
    int n_coins,
    int n_returns
) {
    int coin = blockIdx.x;
    if (coin >= n_coins) return;

    __shared__ double s_sum[256];
    __shared__ double s_var[256];
    int tid = threadIdx.x;

    double local_sum = 0.0;
    for (int j = tid; j < n_returns; j += blockDim.x) {
        local_sum += returns[coin * n_returns + j];
    }
    s_sum[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }

    __shared__ double s_mean;
    if (tid == 0) {
        s_mean = s_sum[0] / n_returns;
        means_out[coin] = s_mean;
    }
    __syncthreads();

    double local_var = 0.0;
    for (int j = tid; j < n_returns; j += blockDim.x) {
        double c = returns[coin * n_returns + j] - s_mean;
        centered[coin * n_returns + j] = c;
        local_var += c * c;
    }
    s_var[tid] = local_var;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_var[tid] += s_var[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        stds_out[coin] = sqrt(s_var[0] / n_returns);
    }
}

__global__ void corr_normalize(
    const double* __restrict__ cov_matrix,
    const double* __restrict__ stds,
    double* __restrict__ corr_matrix,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;

    int i = idx / n;
    int j = idx % n;
    double si = stds[i];
    double sj = stds[j];
    double denom = si * sj * (double)n;
    double val = (denom > 1e-15) ? cov_matrix[idx] / denom : 0.0;
    if (val > 1.0) val = 1.0;
    if (val < -1.0) val = -1.0;
    corr_matrix[idx] = val;
}

__global__ void fingerprint_latest(
    const double* __restrict__ returns,
    double* __restrict__ metrics_out,
    int n_coins,
    int n_returns,
    int btc_idx,
    int eth_idx
) {
    __shared__ double latest[128];
    __shared__ double sorted[128];
    __shared__ double s_rv[256];
    __shared__ double s_scalar[6];
    __shared__ int s_rv_n;

    int tid = threadIdx.x;

    if (tid < n_coins) {
        latest[tid] = returns[tid * n_returns + (n_returns - 1)];
        sorted[tid] = latest[tid];
    }
    __syncthreads();

    if (tid == 0) {
        double sum = 0.0;
        int pos_count = 0;
        for (int i = 0; i < n_coins; i++) {
            sum += latest[i];
            if (latest[i] > 0) pos_count++;
        }
        s_scalar[0] = sum / n_coins;
        s_scalar[1] = (btc_idx < n_coins) ? latest[btc_idx] : 0.0;
        s_scalar[2] = (eth_idx < n_coins) ? latest[eth_idx] : 0.0;
        s_scalar[3] = (double)pos_count / n_coins;

        for (int i = 1; i < n_coins; i++) {
            double key = sorted[i];
            int j = i - 1;
            while (j >= 0 && sorted[j] > key) {
                sorted[j + 1] = sorted[j];
                j--;
            }
            sorted[j + 1] = key;
        }
        int n = n_coins;
        double median = (n % 2) ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
        double iqr = (n >= 4) ? (sorted[3 * n / 4] - sorted[n / 4]) : 0.0;
        s_scalar[4] = median;
        s_scalar[5] = iqr;
        s_rv_n = (n_returns < 512) ? n_returns : 512;
    }
    __syncthreads();

    int rv_n = s_rv_n;
    double local_sum = 0.0;
    for (int t = tid; t < rv_n; t += blockDim.x) {
        double mkt = 0.0;
        for (int c = 0; c < n_coins; c++) mkt += returns[c * n_returns + t];
        mkt /= n_coins;
        local_sum += mkt * mkt;
    }
    s_rv[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_rv[tid] += s_rv[tid + s];
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
            for (int t = 0; t < n_returns; t++) btc_mean += btc_ret[t];
            btc_mean /= n_returns;
            double btc_var = 0.0;
            for (int t = 0; t < n_returns; t++) {
                double d = btc_ret[t] - btc_mean;
                btc_var += d * d;
            }
            double btc_std = sqrt(btc_var / n_returns);

            for (int c = 0; c < n_coins; c++) {
                if (c == btc_idx || c == eth_idx) continue;
                const double* alt_ret = &returns[c * n_returns];
                double alt_mean = 0.0;
                for (int t = 0; t < n_returns; t++) alt_mean += alt_ret[t];
                alt_mean /= n_returns;
                double alt_var = 0.0, cov = 0.0;
                for (int t = 0; t < n_returns; t++) {
                    double db = btc_ret[t] - btc_mean;
                    double da = alt_ret[t] - alt_mean;
                    alt_var += da * da;
                    cov += db * da;
                }
                double alt_std = sqrt(alt_var / n_returns);
                double denom = btc_std * alt_std * n_returns;
                if (denom > 1e-15) {
                    double r = cov / denom;
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

}  // extern "C"
