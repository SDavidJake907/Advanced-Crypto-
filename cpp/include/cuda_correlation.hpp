#pragma once

#include <cstddef>
#include <vector>

// prices is row-major [n_assets x n_points]
std::vector<double> compute_correlation_gpu(
    const std::vector<double>& prices,
    std::size_t n_assets,
    std::size_t n_points
);
