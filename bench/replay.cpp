// Replays a LOBSTER message file through the order book.
//
// Two modes:
//   --bench            : report throughput (msgs/sec) and per-message latency
//                        percentiles for book updates.
//   --dump <out.csv>   : write a top-of-book + order-flow-imbalance feature
//                        file for the research half of the project.
//
// Usage:
//   lob_replay <messages.csv> --bench
//   lob_replay <messages.csv> --dump features.csv

#include "lob/lobster_parser.hpp"
#include "lob/order_book.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace lob;

namespace {

// Order Flow Imbalance (Cont, Kukanov, Stoikov 2014). Defined incrementally
// from successive best bid/ask prices and sizes:
//   e_n = I{Pb_n >= Pb_{n-1}} * Qb_n  - I{Pb_n <= Pb_{n-1}} * Qb_{n-1}
//       - I{Pa_n <= Pa_{n-1}} * Qa_n  + I{Pa_n >= Pa_{n-1}} * Qa_{n-1}
struct OfiState {
    Price pb = 0, pa = 0;
    Quantity qb = 0, qa = 0;
    bool init = false;

    double update(Price nb, Quantity nqb, Price na, Quantity nqa) {
        double e = 0.0;
        if (init) {
            if (nb >= pb) e += nqb;
            if (nb <= pb) e -= qb;
            if (na <= pa) e -= nqa;
            if (na >= pa) e += qa;
        }
        pb = nb; qb = nqb; pa = na; qa = nqa; init = true;
        return e;
    }
};

void run_bench(const std::vector<Message>& msgs, const char* lat_out = nullptr) {
    OrderBook book;
    std::vector<double> lat;
    lat.reserve(msgs.size());

    auto t0 = std::chrono::steady_clock::now();
    for (const auto& m : msgs) {
        auto a = std::chrono::steady_clock::now();
        book.apply(m);
        auto b = std::chrono::steady_clock::now();
        lat.push_back(std::chrono::duration<double, std::nano>(b - a).count());
    }
    auto t1 = std::chrono::steady_clock::now();

    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::sort(lat.begin(), lat.end());
    auto pct = [&](double p) { return lat[static_cast<std::size_t>(p * (lat.size() - 1))]; };

    std::printf("messages        : %zu\n", msgs.size());
    std::printf("wall time       : %.3f s\n", secs);
    std::printf("throughput      : %.2f M msg/s\n", msgs.size() / secs / 1e6);
    std::printf("latency p50      : %.1f ns\n", pct(0.50));
    std::printf("latency p99      : %.1f ns\n", pct(0.99));
    std::printf("latency p99.9    : %.1f ns\n", pct(0.999));
    std::printf("price levels     : %zu\n", book.num_levels());

    if (lat_out) {
        std::FILE* f = std::fopen(lat_out, "wb");
        std::fprintf(f, "ns\n");
        for (double v : lat) std::fprintf(f, "%.1f\n", v);
        std::fclose(f);
        std::printf("wrote latency samples to %s\n", lat_out);
    }
}

// One row of the LOBSTER orderbook file: top-of-book ground truth.
struct ObTop { Price ask, bid; Quantity asz, bsz; };

// Full first snapshot (all levels) used to seed the pre-open book.
struct ObSeed { std::vector<Price> ask_px, bid_px; std::vector<Quantity> ask_sz, bid_sz; };

std::vector<ObTop> load_orderbook_top(const std::string& path, ObSeed* seed = nullptr) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<ObTop> rows;
    long ask, asz, bid, bsz;
    bool first = true;
    while (std::fscanf(f, "%ld,%ld,%ld,%ld", &ask, &asz, &bid, &bsz) == 4) {
        rows.push_back({ask, bid, asz, bsz});
        if (first && seed) {
            // capture every level on the opening line for seeding
            seed->ask_px.push_back(ask); seed->ask_sz.push_back(asz);
            seed->bid_px.push_back(bid); seed->bid_sz.push_back(bsz);
            long a, as, b, bs;
            while (std::fscanf(f, ",%ld,%ld,%ld,%ld", &a, &as, &b, &bs) == 4) {
                if (a > 0)  { seed->ask_px.push_back(a); seed->ask_sz.push_back(as); }
                if (b > 0)  { seed->bid_px.push_back(b); seed->bid_sz.push_back(bs); }
            }
        }
        first = false;
        int ch; while ((ch = std::fgetc(f)) != '\n' && ch != EOF) {}
    }
    std::fclose(f);
    return rows;
}

// Compares the reconstructed book against the exchange-published orderbook file,
// message by message. The initial book is seeded from the first ground-truth
// snapshot (phantom orders) so we start with the pre-open resting depth that the
// message stream alone does not contain.
void run_validate(const std::vector<Message>& msgs, const std::string& ob_path) {
    ObSeed seed;
    auto ob = load_orderbook_top(ob_path, &seed);
    std::size_t n = std::min(msgs.size(), ob.size());

    OrderBook book;
    // Seed the full pre-open depth from the first ground-truth snapshot so that
    // cancels/executions of orders resting before the open hit real levels.
    for (std::size_t k = 0; k < seed.bid_px.size(); ++k)
        book.seed(Side::Buy, seed.bid_px[k], seed.bid_sz[k]);
    for (std::size_t k = 0; k < seed.ask_px.size(); ++k)
        book.seed(Side::Sell, seed.ask_px[k], seed.ask_sz[k]);

    std::size_t price_match = 0, full_match = 0;
    std::size_t first_div = n;
    // The seed reflects the book *after* message 0, so replay from message 1.
    for (std::size_t i = 1; i < n; ++i) {
        book.apply(msgs[i]);
        bool pm = (book.best_bid() == ob[i].bid && book.best_ask() == ob[i].ask);
        bool fm = pm && book.bid_size() == ob[i].bsz && book.ask_size() == ob[i].asz;
        price_match += pm;
        full_match += fm;
        if (!pm && first_div == n) first_div = i;
    }

    std::printf("messages compared : %zu\n", n);
    std::printf("best bid/ask price match : %.2f%%\n", 100.0 * price_match / n);
    std::printf("price + size match       : %.2f%%\n", 100.0 * full_match / n);
    std::printf("first price divergence   : msg %zu\n", first_div);
}

void run_dump(const std::vector<Message>& msgs, const std::string& out) {
    OrderBook book;
    OfiState ofi;
    std::FILE* f = std::fopen(out.c_str(), "wb");
    std::fprintf(f, "time,bid,bid_sz,ask,ask_sz,mid,ofi\n");
    for (const auto& m : msgs) {
        book.apply(m);
        if (book.best_bid() == 0 || book.best_ask() == 0) continue;
        double e = ofi.update(book.best_bid(), book.bid_size(),
                              book.best_ask(), book.ask_size());
        std::fprintf(f, "%.9f,%lld,%lld,%lld,%lld,%.1f,%.1f\n",
                     m.time,
                     (long long)book.best_bid(), (long long)book.bid_size(),
                     (long long)book.best_ask(), (long long)book.ask_size(),
                     book.mid(), e);
    }
    std::fclose(f);
    std::printf("wrote features to %s\n", out.c_str());
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <messages.csv> --bench | --dump <out.csv>\n", argv[0]);
        return 1;
    }
    auto msgs = load_lobster_messages(argv[1]);
    if (std::strcmp(argv[2], "--bench") == 0) {
        run_bench(msgs, argc >= 4 ? argv[3] : nullptr);
    } else if (std::strcmp(argv[2], "--dump") == 0 && argc >= 4) {
        run_dump(msgs, argv[3]);
    } else if (std::strcmp(argv[2], "--validate") == 0 && argc >= 4) {
        run_validate(msgs, argv[3]);
    } else {
        std::fprintf(stderr, "unknown mode\n");
        return 1;
    }
    return 0;
}
