#pragma once

#include <cstddef>
#include <vector>

std::vector<double> compute_atr_gpu(
    const std::vector<double>& highs,
    const std::vector<double>& lows,
    const std::vector<double>& closes,
    std::size_t n_assets,
    std::size_t n_points,
    int lookback
);
