#pragma once

#include <vector>

#include "nn/layer.hpp"
#include "utils.hpp"

namespace nn {
struct NeuralNetwork {
  std::vector<Layer> layers;

  NeuralNetwork(const std::vector<int>& structure, const ActivationFunction& activation);

  std::vector<float> feedforward(const std::vector<float>& input) const;
};
}  // namespace nn
