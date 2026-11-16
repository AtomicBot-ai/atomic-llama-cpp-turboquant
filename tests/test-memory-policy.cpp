#include "testing.h"
#include "../src/llama-memory-policy.h"

using pressure = llama_memory_pressure;
using reason = llama_memory_policy_reason;

static llama_memory_policy_config config() {
    llama_memory_policy_config c;
    c.mode = llama_memory_mode::automatic;
    c.preferred = 1000; c.minimum = 400; c.reserve = 100;
    c.shrink_step = 200; c.grow_step = 100;
    c.interval_ms = 10; c.recovery_ms = 30; c.max_recovery_ms = 120; c.sample_ttl_ms = 20;
    return c;
}

int main(int argc, char ** argv) {
    testing t;
    if (argc > 1) { t.set_filter(argv[1]); }
    t.test("pressure replay preserves critical across coalescing", [&](testing & t) {
        llama_memory_policy p;
        t.assert_true("valid config", p.configure(config(), 0, 300));
        p.observe({pressure::warning, 0, true}, 0);
        t.assert_equal("warning step", p.decide(0, 1000, 950, 300), size_t(800));
        p.observe({pressure::critical, 0, true}, 1);
        p.observe({pressure::normal, 10000, true}, 2);
        t.assert_equal("critical survives normal and bypasses throttle", p.decide(2, 800, 750, 450), size_t(450));
        t.assert_true("live floor respected", p.status().reason == reason::critical_shrink);
        t.assert_equal("no immediate recovery", p.decide(3, 450, 450, 450), size_t(0));
    });
    t.test("pressure replay gates growth on fresh headroom", [&](testing & t) {
        llama_memory_policy p; p.configure(config(), 0, 300);
        for (uint64_t now : {0, 10, 20, 30}) {
            p.observe({pressure::normal, 10000, true}, now);
            t.assert_equal("sustained normal", p.decide(now, 400, 400, 300), size_t(now == 30 ? 500 : 0));
        }
        p.observe({pressure::normal, 150, true}, 40);
        t.assert_equal("headroom includes all unused capacity", p.decide(40, 500, 400, 300), size_t(0));
        t.assert_true("headroom reason", p.status().reason == reason::headroom);
        t.assert_true("fresh headroom visible", p.status().headroom_valid && p.status().headroom == 150);
        p.observe({pressure::normal, 10000, true}, 50);
        p.decide(50, 500, 400, 300);
        t.assert_equal("expired normal cannot grow", p.decide(80, 500, 400, 300), size_t(0));
        t.assert_true("stale reason", p.status().reason == reason::stale_signal);
        t.assert_true("expired headroom marked invalid", !p.status().headroom_valid);
        p.observe({pressure::normal, 10000, true}, 90);
        t.assert_equal("gap restarts recovery window", p.decide(90, 500, 400, 300), size_t(0));
        p.observe({pressure::normal, 0, false}, 100);
        t.assert_equal("unknown headroom cannot grow", p.decide(100, 500, 400, 300), size_t(0));
    });
    t.test("pressure replay backs off after contested growth", [&](testing & t) {
        llama_memory_policy p; p.configure(config(), 0, 300);
        for (uint64_t now : {0, 10, 20, 30}) {
            p.observe({pressure::normal, 10000, true}, now); p.decide(now, 400, 400, 300);
        }
        p.observe({pressure::warning, 0, true}, 31);
        t.assert_equal("warning rate limited", p.decide(31, 500, 500, 300), size_t(0));
        p.observe({pressure::normal, 10000, true}, 40);
        t.assert_equal("pending warning retained", p.decide(40, 500, 500, 300), size_t(400));
        t.assert_equal("doubled recovery delay", p.status().recovery_delay_ms, uint64_t(60));
        for (uint64_t now = 50; now <= 110; now += 10) {
            p.observe({pressure::normal, 10000, true}, now);
            t.assert_equal("backoff window", p.decide(now, 400, 400, 300), size_t(now == 110 ? 500 : 0));
        }
    });
    t.test("pressure replay honors manual override and mode changes", [&](testing & t) {
        llama_memory_policy p; p.configure(config(), 0, 300);
        p.manual(); p.observe({pressure::critical, 0, true}, 1);
        t.assert_equal("manual is never overwritten", p.decide(1, 800, 750, 300), size_t(0));
        t.assert_true("manual status", p.status().reason == reason::manual_override);
        t.assert_true("explicit resume", p.configure(config(), 2, 300));
        t.assert_equal("old pressure is discarded on resume", p.decide(2, 800, 750, 300), size_t(0));
        p.observe({pressure::critical, 0, true}, 3);
        t.assert_equal("fresh pressure after resume", p.decide(3, 800, 750, 300), size_t(400));
        auto fixed = config(); fixed.mode = llama_memory_mode::fixed; p.configure(fixed, 4, 300);
        p.observe({pressure::critical, 0, true}, 5);
        t.assert_equal("fixed does not react", p.decide(5, 800, 750, 300), size_t(0));
    });
    t.test("pressure replay validates floor time and arithmetic", [&](testing & t) {
        llama_memory_policy p;
        t.assert_true("below live floor rejected", !p.configure(config(), 0, 1001));
        p.configure(config(), 10, 300);
        t.assert_true("old sample rejected", !p.observe({pressure::critical, 0, true}, 9));
        p.observe({pressure::critical, 0, true}, 20);
        t.assert_equal("future sample not consumed early", p.decide(19, 800, 750, 300), size_t(0));
        t.assert_equal("floor above preference cannot silently raise cap", p.decide(20, 800, 750, 1100), size_t(0));
        t.assert_true("floor conflict visible", p.status().reason == reason::floor_conflict);
        auto large = config(); large.preferred = SIZE_MAX; large.grow_step = SIZE_MAX; large.reserve = SIZE_MAX;
        p.configure(large, 30, 300);
        for (uint64_t now : {30, 40, 50, 60}) { p.observe({pressure::normal, SIZE_MAX, true}, now); }
        t.assert_equal("reserve arithmetic does not wrap", p.decide(60, 400, 400, 300), size_t(0));
    });
    t.test("pressure replay flapping does not trigger growth", [&](testing & t) {
        llama_memory_policy p; p.configure(config(), 0, 300);
        size_t applied = 1000;
        for (uint64_t now = 0; now < 200; now += 10) {
            p.observe({now % 20 ? pressure::normal : pressure::warning, 10000, true}, now);
            const size_t next = p.decide(now, applied, applied, 300);
            t.assert_true("no grow during flapping", next == 0 || next <= applied);
            if (next) { applied = next; }
        }
        t.assert_equal("bounded at floor", applied, size_t(400));
    });
    return t.summary();
}
