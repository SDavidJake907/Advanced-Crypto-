#pragma once

#include <cstddef>
#include <vector>

struct FeatureConfig {
    int lookback_mom;
    int lookback_vol;
};

struct FeatureOutput {
    std::vector<double> momentum;
    std::vector<double> volatility;
};

FeatureOutput compute_features_gpu(
    const std::vector<double>& prices,
    std::size_t n_assets,
    std::size_t n_points,
    const FeatureConfig& cfg
);

