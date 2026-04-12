#include "evolution/network_cache.hpp"
#include "core/types.hpp"

namespace moonai {

void NetworkCache::assign(uint32_t e, const Genome &genome) {
  if (e >= networks_.size()) {
    networks_.resize(static_cast<std::size_t>(e) + 1);
  }
  networks_[e] = std::make_unique<NeuralNetwork>(genome);
}

NeuralNetwork *NetworkCache::get(uint32_t e) const {
  if (e == INVALID_ENTITY || e >= networks_.size()) {
    return nullptr;
  }
  return networks_[e].get();
}

void NetworkCache::remove(uint32_t e) {
  if (e == INVALID_ENTITY || e >= networks_.size()) {
    return;
  }
  if (static_cast<std::size_t>(e) + 1 == networks_.size()) {
    networks_.pop_back();
    return;
  }
  networks_[e].reset();
}

void NetworkCache::move_entity(uint32_t from, uint32_t to) {
  if (from == to) {
    return;
  }
  if (from == INVALID_ENTITY || from >= networks_.size() || !networks_[from]) {
    return;
  }
  if (to >= networks_.size()) {
    networks_.resize(static_cast<std::size_t>(to) + 1);
  }
  networks_[to] = std::move(networks_[from]);
}

bool NetworkCache::has(uint32_t e) const {
  return get(e) != nullptr;
}

std::vector<float> NetworkCache::activate(uint32_t e, const std::vector<float> &inputs) const {
  auto *network = get(e);
  if (network) {
    return network->activate(inputs);
  }
  return {};
}

void NetworkCache::activate_batch(std::size_t entity_count, const std::vector<float> &all_inputs,
                                  std::vector<float> &all_outputs, int inputs_per_network, int outputs_per_network) {
  all_outputs.resize(entity_count * outputs_per_network);

  const uint32_t count = static_cast<uint32_t>(entity_count);
  for (uint32_t e = 0; e < count; ++e) {
    const std::size_t i = e;
    const float *input_ptr = &all_inputs[i * inputs_per_network];
    float *output_ptr = &all_outputs[i * outputs_per_network];

    auto *network = get(e);
    if (network) {
      network->activate_into(input_ptr, inputs_per_network, output_ptr, outputs_per_network);
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
