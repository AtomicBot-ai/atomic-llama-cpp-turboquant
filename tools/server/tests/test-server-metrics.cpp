#include "server-metrics.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static void expect_equal(const std::string & expected, const std::string & actual) {
    if (expected != actual) {
        std::cerr << "expected:\n" << expected << "actual:\n" << actual;
        throw std::runtime_error("Prometheus metric formatting mismatch");
    }
}

static void expect_integer(const json & value, const std::string & expected) {
    expect_equal(
        "# HELP llamacpp:test_metric Test metric.\n"
        "# TYPE llamacpp:test_metric counter\n"
        "llamacpp:test_metric " + expected + "\n",
        server_prometheus_format_metric("test_metric", "counter", "Test metric.", value));
}

static void expect_float_round_trip(double value) {
    const std::string output = server_prometheus_format_metric(
        "test_gauge", "gauge", "Test gauge.", json(value));
    const std::string prefix = "llamacpp:test_gauge ";
    const size_t sample = output.rfind(prefix);
    if (sample == std::string::npos || output.back() != '\n') {
        throw std::runtime_error("Prometheus sample line is unavailable");
    }
    const std::string formatted = output.substr(
        sample + prefix.size(), output.size() - sample - prefix.size() - 1);
    char * end = nullptr;
    const double reparsed = std::strtod(formatted.c_str(), &end);
    if (end != formatted.c_str() + formatted.size() || reparsed != value) {
        throw std::runtime_error("floating metric did not round trip");
    }
}

static void expect_invalid(const json & value) {
    try {
        (void) server_prometheus_format_metric("invalid", "gauge", "Invalid metric.", value);
    } catch (const std::invalid_argument &) {
        return;
    }
    throw std::runtime_error("invalid Prometheus metric value was accepted");
}

static void test_endpoint_layout_and_regression_values() {
    expect_equal(
        "# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.\n"
        "# TYPE llamacpp:prompt_tokens_total counter\n"
        "llamacpp:prompt_tokens_total 1423152\n",
        server_prometheus_format_metric(
            "prompt_tokens_total",
            "counter",
            "Number of prompt tokens processed.",
            json(uint64_t(1423152))));
    expect_equal(
        "# HELP llamacpp:tokens_predicted_total Number of generation tokens processed.\n"
        "# TYPE llamacpp:tokens_predicted_total counter\n"
        "llamacpp:tokens_predicted_total 180\n",
        server_prometheus_format_metric(
            "tokens_predicted_total",
            "counter",
            "Number of generation tokens processed.",
            json(uint64_t(180))));
}

static void test_integer_boundaries() {
    expect_integer(json(uint64_t(0)), "0");
    expect_integer(json(uint64_t(999999)), "999999");
    expect_integer(json(uint64_t(1000000)), "1000000");
    expect_integer(json(uint64_t(1030220)), "1030220");
    expect_integer(json(uint64_t(1423152)), "1423152");
    expect_integer(json(uint64_t(9007199254740991ULL)), "9007199254740991");
    expect_integer(json(uint64_t(9007199254740992ULL)), "9007199254740992");
    expect_integer(json(uint64_t(9007199254740993ULL)), "9007199254740993");
    expect_integer(json(std::numeric_limits<uint64_t>::max()), "18446744073709551615");
    expect_integer(json(int64_t(-42)), "-42");
    expect_integer(json(std::numeric_limits<int64_t>::min()), "-9223372036854775808");
    expect_integer(json(std::numeric_limits<int64_t>::max()), "9223372036854775807");
}

static void test_floating_gauges() {
    expect_equal(
        "# HELP llamacpp:fractional_gauge Fractional test gauge.\n"
        "# TYPE llamacpp:fractional_gauge gauge\n"
        "llamacpp:fractional_gauge 1.0000000000000002\n",
        server_prometheus_format_metric(
            "fractional_gauge",
            "gauge",
            "Fractional test gauge.",
            json(1.0000000000000002)));
    for (double value : std::vector<double> {
             0.125,
             -12345.678901234567,
             std::numeric_limits<double>::denorm_min(),
             std::numeric_limits<double>::max(),
         }) {
        expect_float_round_trip(value);
    }
}

static void test_json_number_kinds_remain_distinct() {
    expect_integer(json(uint64_t(9007199254740993ULL)), "9007199254740993");
    expect_equal(
        "# HELP llamacpp:test_metric Test metric.\n"
        "# TYPE llamacpp:test_metric gauge\n"
        "llamacpp:test_metric 9007199254740992\n",
        server_prometheus_format_metric(
            "test_metric",
            "gauge",
            "Test metric.",
            json(static_cast<double>(9007199254740993ULL))));
}

static void test_invalid_values() {
    expect_invalid(json(nullptr));
    expect_invalid(json(true));
    expect_invalid(json("1423152"));
    expect_invalid(json::array({1}));
    expect_invalid(json::object({{"value", 1}}));
    expect_invalid(json(std::numeric_limits<double>::quiet_NaN()));
    expect_invalid(json(std::numeric_limits<double>::infinity()));
    expect_invalid(json(-std::numeric_limits<double>::infinity()));
}

int main() {
    test_endpoint_layout_and_regression_values();
    test_integer_boundaries();
    test_floating_gauges();
    test_json_number_kinds_remain_distinct();
    test_invalid_values();
    return 0;
}
