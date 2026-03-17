#include "kraken_ws.hpp"

#include <chrono>
#include <utility>

KrakenWsClient::KrakenWsClient(const std::string& url) : url_(url) {}

KrakenWsClient::~KrakenWsClient() {
    stop();
}

void KrakenWsClient::set_trade_callback(TradeCallback cb) {
    trade_cb_ = std::move(cb);
}

void KrakenWsClient::subscribe_trades(const std::string& symbol) {
    subscribed_symbol_ = symbol;
}

void KrakenWsClient::start() {
    if (running_.exchange(true)) {
        return;
    }
    worker_ = std::thread(&KrakenWsClient::run, this);
}

void KrakenWsClient::stop() {
    if (!running_.exchange(false)) {
        return;
    }
    if (worker_.joinable()) {
        worker_.join();
    }
}

void KrakenWsClient::run() {
    // Stub loop. This keeps the binding buildable until a websocket backend is added.
    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

