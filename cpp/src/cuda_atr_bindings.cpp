#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cuda_atr.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cuda_atr, m) {
    m.def(
        "compute_atr_gpu",
        &compute_atr_gpu,
        py::arg("highs"),
        py::arg("lows"),
        py::arg("closes"),
        py::arg("n_assets"),
        py::arg("n_points"),
        py::arg("lookback")
    );
}
