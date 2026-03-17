#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cuda_correlation.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cuda_correlation, m) {
    m.def(
        "compute_correlation_gpu",
        &compute_correlation_gpu,
        py::arg("prices"),
        py::arg("n_assets"),
        py::arg("n_points")
    );
}
