#pragma once

#include "neural_network.hpp"

namespace nn {
struct Individual {
  NeuralNetwork network;
  float fitness{};

  Individual(const std::vector<int>& structure, const ActivationFunction& activation);

  void mutate(float mutationRate, float mutationStrength);
  Individual crossover(const Individual& partner) const;
};
}  // namespace nn
