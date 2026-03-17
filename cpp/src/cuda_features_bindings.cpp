#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cuda_features.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cuda_features, m) {
    py::class_<FeatureConfig>(m, "FeatureConfig")
        .def(py::init<>())
        .def_readwrite("lookback_mom", &FeatureConfig::lookback_mom)
        .def_readwrite("lookback_vol", &FeatureConfig::lookback_vol);

    py::class_<FeatureOutput>(m, "FeatureOutput")
        .def_readonly("momentum", &FeatureOutput::momentum)
        .def_readonly("volatility", &FeatureOutput::volatility);

    m.def(
        "compute_features_gpu",
        &compute_features_gpu,
        py::arg("prices"),
        py::arg("n_assets"),
        py::arg("n_points"),
        py::arg("cfg")
    );
}

