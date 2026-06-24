#pragma once

#include <cstdint>

namespace lob {

// Prices are stored as integer ticks (LOBSTER uses price * 10000, i.e.
// dollars in units of 1/10000). Keeping them integral avoids floating point
// comparison issues on the hot path.
using Price = std::int64_t;
using Quantity = std::int64_t;
using OrderId = std::uint64_t;

enum class Side : std::uint8_t { Buy, Sell };

// LOBSTER message event types.
// 1: new limit order
// 2: partial cancellation (modification)
// 3: full deletion (cancel)
// 4: execution of a visible limit order
// 5: execution of a hidden limit order
// 6: cross trade (auction) -- ignored for book building
// 7: trading halt -- ignored for book building
enum class EventType : std::uint8_t {
    Add = 1,
    PartialCancel = 2,
    Delete = 3,
    Execute = 4,
    ExecuteHidden = 5,
    Cross = 6,
    Halt = 7,
};

struct Message {
    double time;        // seconds after midnight
    EventType type;
    OrderId id;
    Quantity size;
    Price price;
    Side side;          // direction of the *limit order* that the event acts on
};

} // namespace lob
