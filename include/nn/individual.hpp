#pragma once
#ifndef INDIVIDUAL_HPP
#define INDIVIDUAL_HPP

#include "neural_network.hpp"

namespace nn {
struct Individual {
  NeuralNetwork network;
  float fitness = 0.0f;

  Individual(const std::vector<int>& structure, const ActivationFunction& activation)
      : network(structure, activation) {}

  void mutate(float mutationRate, float mutationStrength) {
    std::random_device rd;
    std::default_random_engine engine(rd());
    std::uniform_real_distribution<float> chance(0.0f, 1.0f);
    std::normal_distribution<float> perturbation(0.0f, mutationStrength);

    for (auto& layer : network.layers) {
      for (auto& row : layer.weights) {
        for (auto& w : row) {
          if (chance(engine) < mutationRate) {
            w += perturbation(engine);
          }
        }
      }

      for (auto& b : layer.bias) {
        if (chance(engine) < mutationRate) {
          b += perturbation(engine);
        }
      }
    }
  }

  Individual crossover(const Individual& partner) const {
    Individual offspring = *this;
    std::random_device rd;
    std::default_random_engine engine(rd());
    std::uniform_int_distribution<int> choose(0, 1);

    for (size_t l = 0; l < offspring.network.layers.size(); ++l) {
      for (size_t i = 0; i < offspring.network.layers[l].weights.size(); ++i) {
        for (size_t j = 0; j < offspring.network.layers[l].weights[i].size(); ++j) {
          if (choose(engine) == 1) {
            offspring.network.layers[l].weights[i][j] = partner.network.layers[l].weights[i][j];
          }
        }
      }

      for (size_t i = 0; i < offspring.network.layers[l].bias.size(); ++i) {
        if (choose(engine) == 1) {
          offspring.network.layers[l].bias[i] = partner.network.layers[l].bias[i];
        }
      }
    }

    return offspring;
  }
};
}  // namespace nn

#endif
