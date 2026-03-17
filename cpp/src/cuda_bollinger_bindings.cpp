#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cuda_bollinger.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cuda_bollinger, m) {
    py::class_<BollingerOutput>(m, "BollingerOutput")
        .def(py::init<>())
        .def_readonly("middle", &BollingerOutput::middle)
        .def_readonly("upper", &BollingerOutput::upper)
        .def_readonly("lower", &BollingerOutput::lower)
        .def_readonly("bandwidth", &BollingerOutput::bandwidth);

    m.def(
        "compute_bollinger_gpu",
        &compute_bollinger_gpu,
        py::arg("prices"),
        py::arg("n_assets"),
        py::arg("n_points"),
        py::arg("lookback"),
        py::arg("num_std")
    );
}
