#include "lob/order_book.hpp"

namespace lob {

void OrderBook::apply(const Message& m) {
    switch (m.type) {
        case EventType::Add:
            add(m);
            break;
        case EventType::PartialCancel:
        case EventType::Delete:
        case EventType::Execute:
            reduce(m);
            break;
        case EventType::ExecuteHidden:
        case EventType::Cross:
        case EventType::Halt:
            // Hidden executions and auctions/halts do not touch the visible
            // book, so building the visible book ignores them.
            break;
    }
}

void OrderBook::reduce(const Message& m) {
    Book& b = book(m.side);
    auto lvl = b.find(m.price);
    if (lvl == b.end()) {
        // No resting quantity at this price in our book. Nothing to remove --
        // robust against deep moves beyond the seeded depth.
        return;
    }
    lvl->second -= (m.size < lvl->second) ? m.size : lvl->second;
    if (lvl->second <= 0) b.erase(lvl);
}

} // namespace lob
