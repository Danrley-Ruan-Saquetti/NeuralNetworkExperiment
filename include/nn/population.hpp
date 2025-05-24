#pragma once

#include <random>

#include "nn/individual.hpp"

namespace nn {
struct Population {
  std::vector<Individual> individuals;
  const std::vector<int> structure;
  const ActivationFunction activation;

  Population(int size, const std::vector<int>& structure, const ActivationFunction& activation);

  const Individual& getBest() const;
  void evolve(float mutationRate, float mutationStrength, int eliteCount);

 private:
  const Individual& tournamentSelect(int k = 3) const;
};
}  // namespace nn
