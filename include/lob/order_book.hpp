#pragma once

#include "lob/types.hpp"

#include <map>

namespace lob {

// L2 (price-level aggregate) limit order book reconstructed from a LOBSTER-style
// message stream.
//
// Design notes (the things an interviewer will ask about):
//   * Bids and asks are kept in ordered std::map keyed by price. Best bid is
//     rbegin() of bids_, best ask is begin() of asks_. O(log L) per level op
//     where L is the number of distinct price levels (small, ~tens).
//   * We key book mutations on the (price, side) carried by *every* LOBSTER
//     message rather than on order id. Cancels and executions therefore work
//     even for orders that rested before our observation window began -- the
//     reason a naive id-keyed reconstruction drifts away from the real book.
//   * Everything is integer-priced; no floating point on the hot path.
class OrderBook {
public:
    void apply(const Message& m);

    // Seed an initial price level (used to load the pre-open book from a
    // ground-truth snapshot before replaying the message stream).
    void seed(Side side, Price price, Quantity qty) {
        if (qty > 0) book(side)[price] += qty;
    }

    // Best prices/quantities. Returns 0 when a side is empty.
    Price best_bid() const { return bids_.empty() ? 0 : bids_.rbegin()->first; }
    Price best_ask() const { return asks_.empty() ? 0 : asks_.begin()->first; }
    Quantity bid_size() const { return bids_.empty() ? 0 : bids_.rbegin()->second; }
    Quantity ask_size() const { return asks_.empty() ? 0 : asks_.begin()->second; }

    // Mid price in ticks; 0 if either side empty.
    double mid() const {
        if (bids_.empty() || asks_.empty()) return 0.0;
        return 0.5 * static_cast<double>(best_bid() + best_ask());
    }

    std::size_t num_levels() const { return bids_.size() + asks_.size(); }

private:
    using Book = std::map<Price, Quantity>;

    Book& book(Side s) { return s == Side::Buy ? bids_ : asks_; }

    void add(const Message& m) { book(m.side)[m.price] += m.size; }
    void reduce(const Message& m);

    // Ascending price -> aggregate resting quantity. Best bid is the largest
    // key, best ask the smallest.
    Book bids_;
    Book asks_;
};

} // namespace lob
