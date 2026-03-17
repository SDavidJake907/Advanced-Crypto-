#pragma once

#include <cstddef>
#include <vector>

struct BollingerOutput {
    std::vector<double> middle;
    std::vector<double> upper;
    std::vector<double> lower;
    std::vector<double> bandwidth;
};

BollingerOutput compute_bollinger_gpu(
    const std::vector<double>& prices,
    std::size_t n_assets,
    std::size_t n_points,
    int lookback,
    double num_std
);
