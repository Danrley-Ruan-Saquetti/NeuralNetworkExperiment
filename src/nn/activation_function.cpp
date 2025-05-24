#include "nn/activation_function.hpp"

#include <cmath>

namespace nn {
ActivationFunction sigmoid = {
    [](float x) { return 1.0f / (1.0f + std::exp(-x)); },
    [](float y) { return y * (1.0f - y); }};

ActivationFunction relu = {
    [](float x) { return x > 0 ? x : 0.0f; },
    [](float y) { return y > 0 ? 1.0f : 0.0f; }};
}
