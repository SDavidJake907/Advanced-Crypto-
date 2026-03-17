#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <thread>

struct TradeEvent {
    std::string symbol;
    double price;
    double volume;
    long long ts_ms;
};

class KrakenWsClient {
public:
    using TradeCallback = std::function<void(const TradeEvent&)>;

    explicit KrakenWsClient(const std::string& url);
    ~KrakenWsClient();

    void set_trade_callback(TradeCallback cb);
    void subscribe_trades(const std::string& symbol);
    void start();
    void stop();

private:
    std::string url_;
    std::string subscribed_symbol_;
    TradeCallback trade_cb_;
    std::thread worker_;
    std::atomic<bool> running_{false};

    void run();
};

