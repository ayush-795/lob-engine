#pragma once

#include "lob/types.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace lob {

// Parses a LOBSTER "message" CSV file.
//
// LOBSTER message file format (one event per line, no header):
//   Time, Type, OrderID, Size, Price, Direction
// where Direction is +1 for a buy limit order, -1 for a sell limit order.
// Reference: https://lobsterdata.com/info/DataStructure.php
//
// We parse with a hand-rolled scanner instead of std::getline/stringstream
// because parsing is on the critical path when streaming tens of millions of
// rows; this keeps allocations out of the loop.
inline std::vector<Message> load_lobster_messages(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);

    std::vector<Message> out;
    std::vector<char> buf(1 << 20);
    std::string line;

    auto parse_line = [](const char* s) -> Message {
        // Six comma-separated fields.
        char* end = nullptr;
        Message m{};
        double t = std::strtod(s, &end);
        m.time = t;                       s = end + 1;
        long type = std::strtol(s, &end, 10); s = end + 1;
        m.type = static_cast<EventType>(type);
        m.id = std::strtoull(s, &end, 10); s = end + 1;
        m.size = std::strtoll(s, &end, 10); s = end + 1;
        m.price = std::strtoll(s, &end, 10); s = end + 1;
        long dir = std::strtol(s, &end, 10);
        m.side = (dir > 0) ? Side::Buy : Side::Sell;
        return m;
    };

    std::size_t n;
    std::string carry;
    while ((n = std::fread(buf.data(), 1, buf.size(), f)) > 0) {
        carry.append(buf.data(), n);
        std::size_t start = 0;
        for (std::size_t i = 0; i < carry.size(); ++i) {
            if (carry[i] == '\n') {
                if (i > start) out.push_back(parse_line(carry.c_str() + start));
                start = i + 1;
            }
        }
        carry.erase(0, start);
    }
    if (!carry.empty()) out.push_back(parse_line(carry.c_str()));

    std::fclose(f);
    return out;
}

} // namespace lob
