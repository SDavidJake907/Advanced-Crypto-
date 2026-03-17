#pragma once

#include <cstddef>
#include <vector>

struct NorthstarBatchFeatureOutput {
    std::vector<double> hurst;
    std::vector<double> entropy;
    std::vector<double> autocorr;
};

struct NorthstarFingerprintOutput {
    std::vector<double> metrics;
};

NorthstarBatchFeatureOutput compute_northstar_batch_features_gpu(
    const std::vector<double>& prices,
    std::size_t n_assets,
    std::size_t n_points
);

NorthstarFingerprintOutput compute_northstar_fingerprint_gpu(
    const std::vector<double>& prices,
    std::size_t n_assets,
    std::size_t n_points,
    int btc_idx,
    int eth_idx
);
