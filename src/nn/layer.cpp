#include "nn/layer.hpp"

#include <random>
#include <vector>

#include "nn/activation_function.hpp"

namespace nn {
Layer::Layer(int inputs, int outputs, ActivationFunction activation) : activation(activation) {
  std::random_device rd;
  std::default_random_engine engine(rd());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  weights.resize(outputs, std::vector<float>(inputs));
  bias.resize(outputs);

  for (auto& line : weights) {
    for (auto& weight : line) {
      weight = dist(engine);
    }
  }

  for (auto& b : bias) {
    b = dist(engine);
  }
}

std::vector<float> Layer::process(const std::vector<float>& inputs) const {
  std::vector<float> outputs(bias.size());

  for (size_t i = 0; i < outputs.size(); ++i) {
    float total = bias[i];

    for (size_t j = 0; j < inputs.size(); ++j) {
      total += weights[i][j] * inputs[j];
    }

    outputs[i] = activation.activate(total);
  }

  return outputs;
}
}  // namespace nn
