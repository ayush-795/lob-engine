// Minimal hand-rolled test harness (no external deps so it builds anywhere).
#include "lob/order_book.hpp"

#include <cstdio>
#include <cstdlib>

using namespace lob;

static int g_failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

static Message add(OrderId id, Side s, Price p, Quantity q) {
    return Message{0.0, EventType::Add, id, q, p, s};
}

static void test_basic_book() {
    OrderBook b;
    b.apply(add(1, Side::Buy, 100, 10));
    b.apply(add(2, Side::Buy, 101, 5));
    b.apply(add(3, Side::Sell, 103, 8));
    b.apply(add(4, Side::Sell, 104, 2));

    CHECK(b.best_bid() == 101);
    CHECK(b.bid_size() == 5);
    CHECK(b.best_ask() == 103);
    CHECK(b.ask_size() == 8);
    CHECK(b.mid() == 102.0);
    CHECK(b.num_levels() == 4);
}

static void test_aggregation_same_level() {
    OrderBook b;
    b.apply(add(1, Side::Buy, 100, 10));
    b.apply(add(2, Side::Buy, 100, 7));
    CHECK(b.best_bid() == 100);
    CHECK(b.bid_size() == 17);
}

static void test_full_delete_pops_level() {
    OrderBook b;
    b.apply(add(1, Side::Buy, 100, 10));
    b.apply(add(2, Side::Buy, 99, 5));
    b.apply(Message{0.0, EventType::Delete, 1, 10, 100, Side::Buy});
    CHECK(b.best_bid() == 99);
    CHECK(b.bid_size() == 5);
    CHECK(b.num_levels() == 1);
}

static void test_partial_cancel() {
    OrderBook b;
    b.apply(add(1, Side::Sell, 105, 10));
    b.apply(Message{0.0, EventType::PartialCancel, 1, 4, 105, Side::Sell});
    CHECK(b.best_ask() == 105);
    CHECK(b.ask_size() == 6);
    CHECK(b.num_levels() == 1);
}

static void test_execution_consumes_liquidity() {
    OrderBook b;
    b.apply(add(1, Side::Sell, 105, 10));
    b.apply(Message{0.0, EventType::Execute, 1, 10, 105, Side::Sell});
    CHECK(b.best_ask() == 0);      // level emptied
    CHECK(b.num_levels() == 0);
}

static void test_reduce_at_empty_price_is_noop() {
    OrderBook b;
    b.apply(add(1, Side::Buy, 100, 10));
    // cancel at a price with no resting quantity -- must not corrupt the book
    b.apply(Message{0.0, EventType::Delete, 999, 5, 95, Side::Buy});
    CHECK(b.best_bid() == 100);
    CHECK(b.bid_size() == 10);
}

static void test_seed_then_reduce_unknown_id() {
    OrderBook b;
    // pre-open depth seeded without order ids
    b.seed(Side::Buy, 100, 50);
    // an execution/cancel of a pre-open order is keyed by price+size, not id
    b.apply(Message{0.0, EventType::Delete, 42, 20, 100, Side::Buy});
    CHECK(b.best_bid() == 100);
    CHECK(b.bid_size() == 30);
}

int main() {
    test_basic_book();
    test_aggregation_same_level();
    test_full_delete_pops_level();
    test_partial_cancel();
    test_execution_consumes_liquidity();
    test_reduce_at_empty_price_is_noop();
    test_seed_then_reduce_unknown_id();

    if (g_failures == 0) {
        std::printf("all tests passed\n");
        return 0;
    }
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
}
