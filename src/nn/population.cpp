#include "nn/population.hpp"

#include <random>
#include <vector>

#include "nn/activation_function.hpp"
#include "nn/individual.hpp"

namespace nn {
Population::Population(int size, const std::vector<int>& structure, const ActivationFunction& activation)
    : structure(structure),
      activation(activation) {
  for (int i = 0; i < size; ++i) {
    individuals.emplace_back(structure, activation);
  }
}

const Individual& Population::getBest() const {
  return *std::max_element(individuals.begin(), individuals.end(), [](const Individual& a, const Individual& b) {
    return a.fitness < b.fitness;
  });
}

void Population::evolve(float mutationRate, float mutationStrength, int eliteCount) {
  std::sort(individuals.begin(), individuals.end(), [](const Individual& a, const Individual& b) {
    return a.fitness > b.fitness;
  });

  std::vector<Individual> nextGen;

  for (int i = 0; i < eliteCount && i < individuals.size(); ++i) {
    nextGen.push_back(individuals[i]);
  }

  while (nextGen.size() < individuals.size()) {
    const Individual& parent1 = tournamentSelect();
    const Individual& parent2 = tournamentSelect();

    Individual child = parent1.crossover(parent2);

    child.mutate(mutationRate, mutationStrength);
    nextGen.push_back(std::move(child));
  }

  individuals = std::move(nextGen);
}

const Individual& Population::tournamentSelect(int k) const {
  std::random_device rd;
  std::default_random_engine engine(rd());
  std::uniform_int_distribution<int> dist(0, individuals.size() - 1);

  const Individual* best = nullptr;

  for (int i = 0; i < k; ++i) {
    const Individual& candidate = individuals[dist(engine)];

    if (!best || candidate.fitness > best->fitness) {
      best = &candidate;
    }
  }
  return *best;
}
}  // namespace nn
