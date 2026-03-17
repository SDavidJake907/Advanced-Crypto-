#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include "execution_engine.hpp"
#include "kraken_ws.hpp"

namespace py = pybind11;

PYBIND11_MODULE(krakencpp, m) {
    py::class_<TradeEvent>(m, "TradeEvent")
        .def(py::init<>())
        .def_readonly("symbol", &TradeEvent::symbol)
        .def_readonly("price", &TradeEvent::price)
        .def_readonly("volume", &TradeEvent::volume)
        .def_readonly("ts_ms", &TradeEvent::ts_ms);

    py::class_<KrakenWsClient>(m, "KrakenWsClient")
        .def(py::init<const std::string&>())
        .def("set_trade_callback", &KrakenWsClient::set_trade_callback)
        .def("subscribe_trades", &KrakenWsClient::subscribe_trades)
        .def("start", &KrakenWsClient::start)
        .def("stop", &KrakenWsClient::stop);

    py::class_<ExecRequest>(m, "ExecRequest")
        .def(py::init<>())
        .def_readwrite("symbol", &ExecRequest::symbol)
        .def_readwrite("side", &ExecRequest::side)
        .def_readwrite("qty", &ExecRequest::qty)
        .def_readwrite("price", &ExecRequest::price);

    py::class_<ExecResult>(m, "ExecResult")
        .def(py::init<>())
        .def_readonly("status", &ExecResult::status)
        .def_readonly("symbol", &ExecResult::symbol)
        .def_readonly("side", &ExecResult::side)
        .def_readonly("qty", &ExecResult::qty)
        .def_readonly("price", &ExecResult::price)
        .def_readonly("notional", &ExecResult::notional)
        .def_readonly("fee", &ExecResult::fee);

    py::class_<ExecutionEngine>(m, "ExecutionEngine")
        .def(py::init<double>(), py::arg("fee_rate") = 0.001)
        .def("simulate", &ExecutionEngine::simulate);
}

