#pragma once

#include "core/types.hpp"
#include "evolution/genome.hpp"
#include "evolution/neural_network.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace moonai {

struct CompiledNetwork {
  int num_inputs = 0;
  int num_outputs = 0;
  int num_nodes = 0;
  std::vector<int> eval_order;
  std::vector<int> conn_from;
  std::vector<float> conn_weights;
  std::vector<int> conn_ptr;
  std::array<int, OUTPUT_COUNT> output_indices{};

  int num_eval() const {
    return static_cast<int>(eval_order.size());
  }

  int num_connections() const {
    return static_cast<int>(conn_from.size());
  }
};

class NetworkCache {
public:
  void assign(uint32_t e, const Genome &genome);

  NeuralNetwork *get(uint32_t e) const;

  NeuralNetwork *get_network(uint32_t e) const {
    return get(e);
  }

  const CompiledNetwork *get_compiled(uint32_t e) const;

  void remove(uint32_t e);
  void move_entity(uint32_t from, uint32_t to);

  bool has(uint32_t e) const;

  void clear();

  std::size_t size() const {
    return networks_.size();
  }
  bool empty() const {
    return networks_.empty();
  }

private:
  std::vector<std::unique_ptr<NeuralNetwork>> networks_;
  std::vector<std::unique_ptr<CompiledNetwork>> compiled_;
};

} // namespace moonai
