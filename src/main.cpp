#define POPULATION_SIZE 100
#define GENERATIONS 30
#define ELITE_COUNT 5
#define MUTATION_RATE 0.1f
#define MUTATION_STRENGTH 0.3f

#include <cmath>
#include <iostream>
#include <vector>

#include "nn/activation_function.hpp"
#include "nn/individual.hpp"
#include "nn/neural_network.hpp"
#include "nn/population.hpp"

float simulate(const nn::NeuralNetwork& network) {
  std::vector<float> input = {1.0f, 1.0f};
  std::vector<float> output = network.feedforward(input);

  return 1.0f - std::abs(1.0f - output[0]);
}

int main() {
  std::vector<int> structure = {2, 4, 1};
  nn::ActivationFunction activation = nn::sigmoid;

  nn::Population population(POPULATION_SIZE, structure, activation);

  for (int generation = 0; generation < GENERATIONS; ++generation) {
    for (auto& individual : population.individuals) {
      individual.fitness = simulate(individual.network);
    }

    const nn::Individual& best = population.getBest();
    std::cout << "Generation " << generation + 1 << " - Best fitness: " << best.fitness << std::endl;

    population.evolve(MUTATION_RATE, MUTATION_STRENGTH, ELITE_COUNT);
  }

  const nn::Individual& finalBest = population.getBest();
  std::vector<float> testInput = {1.0f, 1.0f};

  std::vector<float> result = finalBest.network.feedforward(testInput);

  std::cout << "Best score: " << result[0] << std::endl;

  return 0;
}
