#pragma once

#include <cmath>
#include <functional>

namespace nn {
using ActivationHandler = std::function<float(float)>;

struct ActivationFunction {
  ActivationHandler activate;
  ActivationHandler derivative;
};

extern ActivationFunction sigmoid;
extern ActivationFunction relu;
}  // namespace nn
