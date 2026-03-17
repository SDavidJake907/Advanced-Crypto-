#pragma once

#include <string>

struct ExecRequest {
    std::string symbol;
    std::string side;
    double qty;
    double price;
};

struct ExecResult {
    std::string status;
    std::string symbol;
    std::string side;
    double qty;
    double price;
    double notional;
    double fee;
};

class ExecutionEngine {
public:
    explicit ExecutionEngine(double fee_rate = 0.001);

    ExecResult simulate(const ExecRequest& req) const;

private:
    double fee_rate_;
};

