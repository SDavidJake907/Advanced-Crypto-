#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cuda_rsi.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cuda_rsi, m) {
    m.def(
        "compute_rsi_gpu",
        &compute_rsi_gpu,
        py::arg("prices"),
        py::arg("n_assets"),
        py::arg("n_points"),
        py::arg("lookback")
    );
}
