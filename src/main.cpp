#define POPULATION_SIZE 100
#define GENERATIONS 100
#define ELITE_COUNT 5
#define MUTATION_RATE 0.1f
#define MUTATION_STRENGTH 0.3f

#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>

#include "nn/activation_function.hpp"
#include "nn/individual.hpp"
#include "nn/neural_network.hpp"
#include "nn/population.hpp"

const std::vector<std::pair<std::vector<float>, float>> DATASET = {
    {{0.0f, 0.0f}, 0.0f},
    {{0.0f, 1.0f}, 1.0f},
    {{1.0f, 0.0f}, 1.0f},
    {{1.0f, 1.0f}, 0.0f},
};

float simulate(const nn::Individual& individual, std::vector<float> input) {
  std::vector<float> output = individual.network.feedforward(input);

  return output[0];
}

float evaluate(const nn::Individual& individual) {
  float totalError = 0.0f;
  
  for (const auto& [input, expected] : DATASET) {
    float output = simulate(individual, input);
    float error = expected - output;

    totalError += error * error;
  }

  return 1.0f / (1.0f + totalError);
}

int main() {
  std::vector<int> structure = {2, 4, 1};
  nn::ActivationFunction activation = nn::sigmoid;

  nn::Population population(POPULATION_SIZE, structure, activation);

  for (int generation = 0; generation < GENERATIONS; ++generation) {
    for (auto& individual : population.individuals) {
      individual.fitness = evaluate(individual);
    }

    const nn::Individual& best = population.getBest();
    std::cout << "Generation " << generation + 1 << " - Best fitness: " << best.fitness << std::endl;

    population.evolve(MUTATION_RATE, MUTATION_STRENGTH, ELITE_COUNT);
  }

  const auto& best = population.getBest();

  std::cout << std::endl
            << "Best individual (XOR):" << std::endl;

  for (const auto& [input, expected] : DATASET) {
    float output = best.network.feedforward(input)[0];

    std::cout << "Input: [" << input[0] << ", " << input[1] << "] | Expect: " << expected << " | Result: " << std::fixed << std::setprecision(4) << output << std::endl;
  }

  return 0;
}
