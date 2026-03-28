#pragma once
#include "evolution/genome.hpp"
#include "evolution/neural_network.hpp"
#include "simulation/entity.hpp"
#include <memory>
#include <unordered_map>
#include <vector>

namespace moonai {

class NetworkCache {
public:
  void assign(Entity e, const Genome &genome,
              const std::string &activation_func);

  NeuralNetwork *get(Entity e) const;

  NeuralNetwork *get_network(Entity e) const {
    return get(e);
  }

  void remove(Entity e);

  bool has(Entity e) const;

  std::vector<float> activate(Entity e, const std::vector<float> &inputs) const;

  void activate_batch(const std::vector<Entity> &entities,
                      const std::vector<float> &all_inputs,
                      std::vector<float> &all_outputs, int inputs_per_network,
                      int outputs_per_network);

  void clear();

  size_t size() const {
    return networks_.size();
  }
  bool empty() const {
    return networks_.empty();
  }

private:
  std::unordered_map<Entity, std::unique_ptr<NeuralNetwork>, EntityHash>
      networks_;
};

} // namespace moonai
