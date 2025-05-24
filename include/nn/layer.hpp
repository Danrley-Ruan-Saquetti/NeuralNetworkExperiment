#pragma once

#include <random>

#include "nn/activation_function.hpp"
#include "utils.hpp"

namespace nn {
struct Layer {
  util::matriz<float> weights;
  std::vector<float> bias;

  ActivationFunction activation;

  Layer(int inputs, int outputs, ActivationFunction activation);

  std::vector<float> process(const std::vector<float>& inputs) const;
};
}  // namespace nn
