#pragma once

#include "core/config.hpp"
#include "core/random.hpp"
#include "evolution/genome.hpp"

#include <cstdint>
#include <map>
#include <utility>

namespace moonai {

class InnovationTracker {
public:
  InnovationTracker() = default;

  // Initialize counters from an existing genome population
  void init_from_population(const std::vector<Genome> &population);

  std::uint32_t get_innovation(std::uint32_t in_node, std::uint32_t out_node);
  std::uint32_t get_split_node_id(std::uint32_t in_node, std::uint32_t out_node);

  std::uint32_t next_node_id();

  void set_counters(std::uint32_t innov, std::uint32_t node) {
    innovation_counter_ = innov;
    node_counter_ = node;
  }

private:
  std::uint32_t innovation_counter_ = 0;
  std::uint32_t node_counter_ = 0;
  std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> innovation_cache_;
  std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> split_node_cache_;
};

class Mutation {
public:
  static void mutate_weights(Genome &genome, Random &rng, float power);
  static void add_connection(Genome &genome, Random &rng, InnovationTracker &tracker);
  static void add_node(Genome &genome, Random &rng, InnovationTracker &tracker, int max_hidden_nodes = 0);
  static void delete_connection(Genome &genome, Random &rng);

  static void mutate(Genome &genome, Random &rng, const SimulationConfig &config, InnovationTracker &tracker);
};

} // namespace moonai
