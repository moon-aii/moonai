#include "evolution/network_cache.hpp"
#include "core/types.hpp"

#include <algorithm>

namespace moonai {

namespace {

CompiledNetwork compile_network(const NeuralNetwork &network) {
  CompiledNetwork compiled;
  compiled.num_inputs = network.num_inputs();
  compiled.num_outputs = network.num_outputs();
  compiled.num_nodes = network.num_nodes();
  compiled.eval_order = network.eval_order_indices();
  compiled.conn_ptr.reserve(compiled.eval_order.size() + 1);

  const auto &incoming = network.incoming();
  int ptr = 0;
  for (const int node_idx : compiled.eval_order) {
    compiled.conn_ptr.push_back(ptr);
    for (const auto &[from_idx, weight] : incoming[static_cast<std::size_t>(node_idx)]) {
      compiled.conn_from.push_back(from_idx);
      compiled.conn_weights.push_back(weight);
      ++ptr;
    }
  }
  compiled.conn_ptr.push_back(ptr);

  compiled.output_indices.fill(0);
  const auto &output_indices = network.output_indices();
  const std::size_t output_count = std::min(output_indices.size(), compiled.output_indices.size());
  for (std::size_t i = 0; i < output_count; ++i) {
    compiled.output_indices[i] = output_indices[i];
  }

  return compiled;
}

} // namespace

void NetworkCache::assign(uint32_t e, const Genome &genome) {
  if (e >= networks_.size()) {
    networks_.resize(static_cast<std::size_t>(e) + 1);
    compiled_.resize(static_cast<std::size_t>(e) + 1);
  }
  networks_[e] = std::make_unique<NeuralNetwork>(genome);
  compiled_[e] = std::make_unique<CompiledNetwork>(compile_network(*networks_[e]));
}

NeuralNetwork *NetworkCache::get(uint32_t e) const {
  if (e == INVALID_ENTITY || e >= networks_.size()) {
    return nullptr;
  }
  return networks_[e].get();
}

const CompiledNetwork *NetworkCache::get_compiled(uint32_t e) const {
  if (e == INVALID_ENTITY || e >= compiled_.size()) {
    return nullptr;
  }
  return compiled_[e].get();
}

void NetworkCache::remove(uint32_t e) {
  if (e == INVALID_ENTITY || e >= networks_.size()) {
    return;
  }
  if (static_cast<std::size_t>(e) + 1 == networks_.size()) {
    networks_.pop_back();
    compiled_.pop_back();
    return;
  }
  networks_[e].reset();
  compiled_[e].reset();
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
    compiled_.resize(static_cast<std::size_t>(to) + 1);
  }
  networks_[to] = std::move(networks_[from]);
  compiled_[to] = std::move(compiled_[from]);
}

bool NetworkCache::has(uint32_t e) const {
  return get(e) != nullptr;
}

void NetworkCache::clear() {
  networks_.clear();
  compiled_.clear();
}

} // namespace moonai
