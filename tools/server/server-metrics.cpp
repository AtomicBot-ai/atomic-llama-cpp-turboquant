#include "server-metrics.h"

#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

std::string server_prometheus_format_metric(
        const std::string & name,
        const std::string & type,
        const std::string & help,
        const json & value) {
    std::string numeric_value;
    if (value.is_number_unsigned()) {
        numeric_value = std::to_string(value.get<uint64_t>());
    } else if (value.is_number_integer()) {
        numeric_value = std::to_string(value.get<int64_t>());
    } else if (value.is_number_float()) {
        const double number = value.get<double>();
        if (!std::isfinite(number)) {
            throw std::invalid_argument("Prometheus metric value is not finite");
        }
        std::ostringstream stream;
        stream.imbue(std::locale::classic());
        stream << std::setprecision(std::numeric_limits<double>::max_digits10) << number;
        if (!stream) {
            throw std::runtime_error("Prometheus metric value could not be formatted");
        }
        numeric_value = stream.str();
    } else {
        throw std::invalid_argument("Prometheus metric value is not numeric");
    }

    const std::string full_name = "llamacpp:" + name;
    return "# HELP " + full_name + " " + help + "\n"
         + "# TYPE " + full_name + " " + type + "\n"
         + full_name + " " + numeric_value + "\n";
}
