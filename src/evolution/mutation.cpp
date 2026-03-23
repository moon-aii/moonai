#include "evolution/mutation.hpp"

#include <algorithm>
#include <stack>
#include <unordered_set>

namespace moonai {

void InnovationTracker::init_from_population(const std::vector<Genome>& population) {
    // Find max innovation number and max node id across all genomes
    for (const auto& genome : population) {
        for (const auto& conn : genome.connections()) {
            if (conn.innovation >= innovation_counter_) {
                innovation_counter_ = conn.innovation + 1;
            }
        }
        for (const auto& node : genome.nodes()) {
            if (node.id >= node_counter_) {
                node_counter_ = node.id + 1;
            }
        }
    }
}

std::uint32_t InnovationTracker::get_innovation(std::uint32_t in_node, std::uint32_t out_node) {
    auto key = std::make_pair(in_node, out_node);
    auto it = generation_innovations_.find(key);
    if (it != generation_innovations_.end()) {
        return it->second;
    }
    std::uint32_t innov = innovation_counter_++;
    generation_innovations_[key] = innov;
    return innov;
}

std::uint32_t InnovationTracker::next_node_id() {
    return node_counter_++;
}

void InnovationTracker::reset_generation() {
    generation_innovations_.clear();
}

void Mutation::mutate_weights(Genome& genome, Random& rng, float power) {
    for (auto& conn : genome.connections()) {
        if (rng.next_bool(0.9f)) {
            // Perturb
            conn.weight += rng.next_gaussian(0.0f, power);
            // Clamp to reasonable range
            conn.weight = std::clamp(conn.weight, -8.0f, 8.0f);
        } else {
            // Replace with random
            conn.weight = rng.next_float(-2.0f, 2.0f);
        }
    }
}

void Mutation::add_connection(Genome& genome, Random& rng, InnovationTracker& tracker) {
    const auto& nodes = genome.nodes();
    if (nodes.size() < 2) return;

    // Cycle detection: DFS from 'to_id' looking for 'from_id'.
    // If found, adding from->to would create a cycle (recurrent connection).
    auto would_create_cycle = [&](std::uint32_t from_id, std::uint32_t to_id) -> bool {
        std::unordered_set<std::uint32_t> visited;
        std::stack<std::uint32_t> stack;
        stack.push(to_id);
        while (!stack.empty()) {
            auto nid = stack.top(); stack.pop();
            if (nid == from_id) return true;
            if (!visited.insert(nid).second) continue;
            for (const auto& c : genome.connections()) {
                if (c.enabled && c.in_node == nid) stack.push(c.out_node);
            }
        }
        return false;
    };

    for (int attempt = 0; attempt < 30; ++attempt) {
        int from_idx = rng.next_int(0, static_cast<int>(nodes.size()) - 1);
        int to_idx = rng.next_int(0, static_cast<int>(nodes.size()) - 1);

        const auto& from = nodes[from_idx];
        const auto& to = nodes[to_idx];

        // Don't connect to input/bias nodes
        if (to.type == NodeType::Input || to.type == NodeType::Bias) continue;
        // Don't connect from output nodes
        if (from.type == NodeType::Output) continue;
        // No self-connections
        if (from.id == to.id) continue;

        // Check if connection already exists
        if (genome.has_connection(from.id, to.id)) continue;

        // Reject cycle-creating connections to keep networks feed-forward
        if (would_create_cycle(from.id, to.id)) continue;

        std::uint32_t innov = tracker.get_innovation(from.id, to.id);
        genome.add_connection({from.id, to.id,
                               rng.next_float(-1.0f, 1.0f),
                               true, innov});
        return;
    }
}

void Mutation::add_node(Genome& genome, Random& rng, InnovationTracker& tracker,
                        int max_hidden_nodes) {
    // Enforce cap on hidden node count (0 = unlimited)
    if (max_hidden_nodes > 0) {
        int hidden_count = 0;
        for (const auto& n : genome.nodes())
            if (n.type == NodeType::Hidden) ++hidden_count;
        if (hidden_count >= max_hidden_nodes) return;
    }

    auto& conns = genome.connections();
    if (conns.empty()) return;

    // Pick a random enabled connection
    std::vector<int> enabled_indices;
    for (int i = 0; i < static_cast<int>(conns.size()); ++i) {
        if (conns[i].enabled) enabled_indices.push_back(i);
    }
    if (enabled_indices.empty()) return;

    int idx = enabled_indices[rng.next_int(0, static_cast<int>(enabled_indices.size()) - 1)];

    // Copy out fields before any add_connection/add_node call that could reallocate conns
    std::uint32_t in_id  = conns[idx].in_node;
    std::uint32_t out_id = conns[idx].out_node;
    float old_weight     = conns[idx].weight;
    conns[idx].enabled   = false;

    std::uint32_t new_id = tracker.next_node_id();
    genome.add_node({new_id, NodeType::Hidden});

    // Connection from original source to new node with weight 1.0
    std::uint32_t innov1 = tracker.get_innovation(in_id, new_id);
    genome.add_connection({in_id, new_id, 1.0f, true, innov1});

    // Connection from new node to original target with original weight
    std::uint32_t innov2 = tracker.get_innovation(new_id, out_id);
    genome.add_connection({new_id, out_id, old_weight, true, innov2});
}

void Mutation::toggle_connection(Genome& genome, Random& rng) {
    auto& conns = genome.connections();
    if (conns.empty()) return;

    int idx = rng.next_int(0, static_cast<int>(conns.size()) - 1);
    conns[idx].enabled = !conns[idx].enabled;
}

void Mutation::delete_connection(Genome& genome, Random& rng) {
    auto& conns = genome.connections();
    if (conns.size() <= 1) return;  // Keep at least one connection

    int idx = rng.next_int(0, static_cast<int>(conns.size()) - 1);
    conns.erase(conns.begin() + idx);
}

void Mutation::mutate(Genome& genome, Random& rng,
                      const SimulationConfig& config,
                      InnovationTracker& tracker) {
    if (rng.next_bool(config.mutation_rate)) {
        mutate_weights(genome, rng, config.weight_mutation_power);
    }
    if (rng.next_bool(config.add_connection_rate)) {
        add_connection(genome, rng, tracker);
    }
    if (rng.next_bool(config.add_node_rate)) {
        add_node(genome, rng, tracker, config.max_hidden_nodes);
    }
    if (rng.next_bool(config.delete_connection_rate)) {
        delete_connection(genome, rng);
    }

    // Safety: ensure at least one connection is enabled to prevent degenerate networks
    bool any_enabled = false;
    for (const auto& conn : genome.connections()) {
        if (conn.enabled) { any_enabled = true; break; }
    }
    if (!any_enabled && !genome.connections().empty()) {
        int idx = rng.next_int(0, static_cast<int>(genome.connections().size()) - 1);
        genome.connections()[idx].enabled = true;
    }
}

} // namespace moonai
