#include "execution_engine.hpp"

ExecutionEngine::ExecutionEngine(double fee_rate) : fee_rate_(fee_rate) {}

ExecResult ExecutionEngine::simulate(const ExecRequest& req) const {
    ExecResult res{};
    if (req.qty <= 0 || req.price <= 0) {
        res.status = "rejected";
        res.symbol = req.symbol;
        res.side = req.side;
        return res;
    }

    const double notional = req.qty * req.price;
    const double fee = notional * fee_rate_;

    res.status = "filled";
    res.symbol = req.symbol;
    res.side = req.side;
    res.qty = req.qty;
    res.price = req.price;
    res.notional = notional;
    res.fee = fee;
    return res;
}

