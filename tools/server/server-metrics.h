#pragma once

#include "server-common.h"

#include <string>

// Format one complete Prometheus metric. Throws std::invalid_argument when
// value is not a supported finite numeric JSON value, or std::runtime_error
// if the value cannot be formatted.
std::string server_prometheus_format_metric(
        const std::string & name,
        const std::string & type,
        const std::string & help,
        const json & value);
