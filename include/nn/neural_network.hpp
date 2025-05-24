#pragma once
#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>

#include "layer.hpp"
#include "utils.hpp"

namespace nn {
struct NeuralNetwork {
  std::vector<Layer> layers;

  NeuralNetwork(const std::vector<int>& structure, const ActivationFunction& activation) {
    for (size_t i = 1; i < structure.size(); ++i) {
      layers.emplace_back(structure[i - 1], structure[i], activation);
    }
  }

  std::vector<float> feedforward(const std::vector<float>& input) const {
    std::vector<float> current = input;

    for (const auto& layer : layers) {
      current = layer.process(current);
    }

    return current;
  }
};
}  // namespace nn
#endif
