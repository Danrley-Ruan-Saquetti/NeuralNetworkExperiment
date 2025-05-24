#include "nn/neural_network.hpp"

#include <vector>

#include "nn/activation_function.hpp"

namespace nn {
NeuralNetwork::NeuralNetwork(const std::vector<int>& structure, const ActivationFunction& activation) {
  for (size_t i = 1; i < structure.size(); ++i) {
    layers.emplace_back(structure[i - 1], structure[i], activation);
  }
}

std::vector<float> NeuralNetwork::feedforward(const std::vector<float>& input) const {
  std::vector<float> current = input;

  for (const auto& layer : layers) {
    current = layer.process(current);
  }

  return current;
}
}  // namespace nn
