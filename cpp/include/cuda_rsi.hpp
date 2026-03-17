#pragma once

#include <cstddef>
#include <vector>

std::vector<double> compute_rsi_gpu(
    const std::vector<double>& prices,
    std::size_t n_assets,
    std::size_t n_points,
    int lookback
);
