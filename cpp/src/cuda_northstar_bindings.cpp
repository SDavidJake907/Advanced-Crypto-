#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cuda_northstar.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cuda_northstar, m) {
    py::class_<NorthstarBatchFeatureOutput>(m, "NorthstarBatchFeatureOutput")
        .def_readonly("hurst", &NorthstarBatchFeatureOutput::hurst)
        .def_readonly("entropy", &NorthstarBatchFeatureOutput::entropy)
        .def_readonly("autocorr", &NorthstarBatchFeatureOutput::autocorr);

    py::class_<NorthstarFingerprintOutput>(m, "NorthstarFingerprintOutput")
        .def_readonly("metrics", &NorthstarFingerprintOutput::metrics);

    m.def(
        "compute_northstar_batch_features_gpu",
        &compute_northstar_batch_features_gpu,
        py::arg("prices"),
        py::arg("n_assets"),
        py::arg("n_points")
    );

    m.def(
        "compute_northstar_fingerprint_gpu",
        &compute_northstar_fingerprint_gpu,
        py::arg("prices"),
        py::arg("n_assets"),
        py::arg("n_points"),
        py::arg("btc_idx"),
        py::arg("eth_idx")
    );
}
