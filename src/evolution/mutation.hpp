#pragma once

#include "evolution/genome.hpp"
#include "core/random.hpp"
#include "core/config.hpp"

#include <map>
#include <cstdint>
#include <utility>

namespace moonai {

// Tracks innovation numbers across a generation to prevent duplicate innovations
class InnovationTracker {
public:
    InnovationTracker() = default;

    // Initialize counters from an existing genome population
    void init_from_population(const std::vector<Genome>& population);

    // Get or create innovation number for a connection (in_node, out_node)
    // Returns the same innovation number if the same structural mutation
    // has already occurred this generation
    std::uint32_t get_innovation(std::uint32_t in_node, std::uint32_t out_node);

    std::uint32_t next_node_id();

    // Reset the per-generation lookup table (call at the start of each generation)
    void reset_generation();

    std::uint32_t innovation_count() const { return innovation_counter_; }
    std::uint32_t node_count() const { return node_counter_; }

    void set_counters(std::uint32_t innov, std::uint32_t node) {
        innovation_counter_ = innov;
        node_counter_ = node;
    }

private:
    std::uint32_t innovation_counter_ = 0;
    std::uint32_t node_counter_ = 0;
    // Per-generation lookup: (in_node, out_node) -> innovation number
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> generation_innovations_;
};

class Mutation {
public:
    static void mutate_weights(Genome& genome, Random& rng, float power);
    static void add_connection(Genome& genome, Random& rng, InnovationTracker& tracker);
    static void add_node(Genome& genome, Random& rng, InnovationTracker& tracker,
                         int max_hidden_nodes = 0);
    static void toggle_connection(Genome& genome, Random& rng);
    static void delete_connection(Genome& genome, Random& rng);

    static void mutate(Genome& genome, Random& rng,
                       const SimulationConfig& config,
                       InnovationTracker& tracker);
};

} // namespace moonai
