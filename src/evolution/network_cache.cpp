#include "evolution/network_cache.hpp"
#include <algorithm>
#include <unordered_set>

namespace moonai {

void NetworkCache::assign(Entity e, const Genome &genome,
                          const std::string &activation_func) {
  networks_[e] = std::make_unique<NeuralNetwork>(genome, activation_func);
}

NeuralNetwork *NetworkCache::get(Entity e) const {
  auto it = networks_.find(e);
  if (it != networks_.end()) {
    return it->second.get();
  }
  return nullptr;
}

void NetworkCache::remove(Entity e) {
  networks_.erase(e);
}

bool NetworkCache::has(Entity e) const {
  return networks_.find(e) != networks_.end();
}

std::vector<float>
NetworkCache::activate(Entity e, const std::vector<float> &inputs) const {
  auto *network = get(e);
  if (network) {
    return network->activate(inputs);
  }
  return {};
}

void NetworkCache::activate_batch(const std::vector<Entity> &entities,
                                  const std::vector<float> &all_inputs,
                                  std::vector<float> &all_outputs,
                                  int inputs_per_network,
                                  int outputs_per_network) {
  all_outputs.resize(entities.size() * outputs_per_network);

  for (size_t i = 0; i < entities.size(); ++i) {
    Entity e = entities[i];
    const float *input_ptr = &all_inputs[i * inputs_per_network];
    float *output_ptr = &all_outputs[i * outputs_per_network];

    auto *network = get(e);
    if (network) {
      network->activate_into(input_ptr, inputs_per_network, output_ptr,
                             outputs_per_network);
    } else {
      // No network found - output zeros
      std::fill(output_ptr, output_ptr + outputs_per_network, 0.0f);
    }
  }
}

void NetworkCache::clear() {
  networks_.clear();
}

} // namespace moonai