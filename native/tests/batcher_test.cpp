// The batcher's contract, stated as tests rather than as comments.
//
// Two properties matter and both are easy to break by "optimising": it is
// work-conserving (the timer bounds how long we wait for *future* arrivals,
// never how long an already-queued request sits), and it has no minimum batch
// size (a lane has at most one request outstanding, so a floor above the number
// of lanes in flight would deadlock).
#include <chrono>
#include <thread>
#include <vector>

#include "check.hpp"
#include "soo/batcher.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

}  // namespace

int main() {
    // With no patience configured, a lone request is dispatched immediately.
    // There is no minimum batch size and there must not be one: a lane has at
    // most one request outstanding, so a floor above the number of lanes in
    // flight would deadlock.
    {
        soo::Batcher batcher(32, 0);
        batcher.submit(7);
        std::vector<int> batch;
        const auto start = Clock::now();
        CHECK(batcher.collect(batch));
        CHECK_EQ(batch.size(), static_cast<std::size_t>(1));
        CHECK_EQ(batch[0], 7);
        CHECK(elapsed_ms(start) < 25.0);
    }

    // With patience configured, the window bounds how long the batcher waits
    // for further arrivals -- it is a bound, not a sleep: the batch is
    // dispatched at the deadline whatever else showed up.
    {
        soo::Batcher batcher(32, 30'000);  // 30 ms
        batcher.submit(1);
        std::vector<int> batch;
        const auto start = Clock::now();
        CHECK(batcher.collect(batch));
        const double waited = elapsed_ms(start);
        CHECK_EQ(batch.size(), static_cast<std::size_t>(1));
        CHECK(waited < 300.0);
    }

    // Batches are capped at max_batch and preserve arrival order; the rest
    // stays queued for the next collect.
    {
        soo::Batcher batcher(4, 0);
        for (int lane = 0; lane < 10; ++lane) batcher.submit(lane);

        std::vector<int> batch;
        CHECK(batcher.collect(batch));
        CHECK_EQ(batch.size(), static_cast<std::size_t>(4));
        for (int i = 0; i < 4; ++i) CHECK_EQ(batch[i], i);

        CHECK(batcher.collect(batch));
        CHECK_EQ(batch.size(), static_cast<std::size_t>(4));
        for (int i = 0; i < 4; ++i) CHECK_EQ(batch[i], i + 4);

        CHECK_EQ(batcher.depth(), static_cast<std::size_t>(2));
        CHECK(batcher.collect(batch));
        CHECK_EQ(batch.size(), static_cast<std::size_t>(2));
        CHECK_EQ(batcher.depth(), static_cast<std::size_t>(0));
    }

    // collect() blocks until something arrives, and picks it up.
    {
        soo::Batcher batcher(32, 1000);
        std::thread producer([&batcher] {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            batcher.submit(3);
        });
        std::vector<int> batch;
        CHECK(batcher.collect(batch));
        CHECK_EQ(batch.size(), static_cast<std::size_t>(1));
        CHECK_EQ(batch[0], 3);
        producer.join();
    }

    // stop() drains what is queued before it reports done. Dropping queued
    // requests here would hang every lane still waiting on one.
    {
        soo::Batcher batcher(2, 0);
        batcher.submit(1);
        batcher.submit(2);
        batcher.submit(3);
        batcher.stop();

        std::vector<int> batch;
        CHECK(batcher.collect(batch));
        CHECK_EQ(batch.size(), static_cast<std::size_t>(2));
        CHECK(batcher.collect(batch));
        CHECK_EQ(batch.size(), static_cast<std::size_t>(1));
        CHECK(!batcher.collect(batch));
    }

    // A stopped, empty batcher never blocks a waiting worker.
    {
        soo::Batcher batcher(8, 100'000);
        batcher.stop();
        std::vector<int> batch;
        const auto start = Clock::now();
        CHECK(!batcher.collect(batch));
        CHECK(elapsed_ms(start) < 25.0);
    }

    return soo_test::report("batcher_test");
}
