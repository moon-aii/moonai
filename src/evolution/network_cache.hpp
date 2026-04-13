#pragma once

#include "evolution/genome.hpp"
#include "evolution/neural_network.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace moonai {

class NetworkCache {
public:
  void assign(uint32_t e, const Genome &genome);

  NeuralNetwork *get(uint32_t e) const;

  NeuralNetwork *get_network(uint32_t e) const {
    return get(e);
  }

  void remove(uint32_t e);
  void move_entity(uint32_t from, uint32_t to);

  bool has(uint32_t e) const;

  std::vector<float> activate(uint32_t e, const std::vector<float> &inputs) const;

  void clear();

  std::size_t size() const {
    return networks_.size();
  }
  bool empty() const {
    return networks_.empty();
  }

private:
  std::vector<std::unique_ptr<NeuralNetwork>> networks_;
};

} // namespace moonai
