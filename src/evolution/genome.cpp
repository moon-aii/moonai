#include "evolution/genome.hpp"

#include "core/profiler.hpp"

#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>

namespace moonai {

Genome::Genome(int num_inputs, int num_outputs)
    : num_inputs_(num_inputs)
    , num_outputs_(num_outputs) {
    std::uint32_t id = 0;
    for (int i = 0; i < num_inputs; ++i) {
        nodes_.push_back({id++, NodeType::Input});
    }
    // Bias node
    nodes_.push_back({id++, NodeType::Bias});
    // Output nodes
    for (int i = 0; i < num_outputs; ++i) {
        nodes_.push_back({id++, NodeType::Output});
    }
}

void Genome::add_node(const NodeGene& node) {
    nodes_.push_back(node);
}

void Genome::add_connection(const ConnectionGene& conn) {
    connections_.push_back(conn);
}

bool Genome::has_connection(std::uint32_t from, std::uint32_t to) const {
    for (const auto& c : connections_) {
        if (c.in_node == from && c.out_node == to) return true;
    }
    return false;
}

bool Genome::has_node(std::uint32_t id) const {
    for (const auto& n : nodes_) {
        if (n.id == id) return true;
    }
    return false;
}

std::uint32_t Genome::max_node_id() const {
    std::uint32_t max_id = 0;
    for (const auto& n : nodes_) {
        if (n.id > max_id) max_id = n.id;
    }
    return max_id;
}

int Genome::complexity() const {
    return static_cast<int>(nodes_.size() + connections_.size());
}

void Genome::sort_connections_by_innovation() {
    std::sort(connections_.begin(), connections_.end(),
              [](const ConnectionGene& lhs, const ConnectionGene& rhs) {
                  return lhs.innovation < rhs.innovation;
              });
}

float Genome::compatibility_distance(const Genome& a, const Genome& b,
                                     float c1, float c2, float c3) {
    MOONAI_PROFILE_INC(ProfileCounter::CompatibilityChecks);
    const auto& raw_conns_a = a.connections();
    const auto& raw_conns_b = b.connections();

    auto by_innovation = [](const ConnectionGene& lhs, const ConnectionGene& rhs) {
        return lhs.innovation < rhs.innovation;
    };
    auto is_sorted_by_innovation = [](const std::vector<ConnectionGene>& conns) {
        for (std::size_t idx = 1; idx < conns.size(); ++idx) {
            if (conns[idx - 1].innovation > conns[idx].innovation) {
                return false;
            }
        }
        return true;
    };
    const bool sorted_a = is_sorted_by_innovation(raw_conns_a);
    const bool sorted_b = is_sorted_by_innovation(raw_conns_b);

    std::vector<ConnectionGene> sorted_copy_a;
    std::vector<ConnectionGene> sorted_copy_b;
    if (!sorted_a) {
        sorted_copy_a = raw_conns_a;
        std::sort(sorted_copy_a.begin(), sorted_copy_a.end(), by_innovation);
    }
    if (!sorted_b) {
        sorted_copy_b = raw_conns_b;
        std::sort(sorted_copy_b.begin(), sorted_copy_b.end(), by_innovation);
    }

    const auto& conns_a = sorted_a ? raw_conns_a : sorted_copy_a;
    const auto& conns_b = sorted_b ? raw_conns_b : sorted_copy_b;

    if (conns_a.empty() && conns_b.empty()) return 0.0f;

    int excess = 0, disjoint = 0, matching = 0;
    float weight_diff = 0.0f;

    std::size_t i = 0;
    std::size_t j = 0;
    while (i < conns_a.size() && j < conns_b.size()) {
        const auto innov_a = conns_a[i].innovation;
        const auto innov_b = conns_b[j].innovation;
        if (innov_a == innov_b) {
            ++matching;
            weight_diff += std::abs(conns_a[i].weight - conns_b[j].weight);
            ++i;
            ++j;
        } else if (innov_a < innov_b) {
            ++disjoint;
            ++i;
        } else {
            ++disjoint;
            ++j;
        }
    }

    const auto max_a = conns_a.empty() ? 0U : conns_a.back().innovation;
    const auto max_b = conns_b.empty() ? 0U : conns_b.back().innovation;
    const auto min_max = std::min(max_a, max_b);

    for (; i < conns_a.size(); ++i) {
        if (conns_a[i].innovation > min_max) {
            ++excess;
        } else {
            ++disjoint;
        }
    }
    for (; j < conns_b.size(); ++j) {
        if (conns_b[j].innovation > min_max) {
            ++excess;
        } else {
            ++disjoint;
        }
    }

    float avg_weight = matching > 0 ? weight_diff / matching : 0.0f;
    float n = static_cast<float>(std::max(conns_a.size(), conns_b.size()));
    if (n < 1.0f) n = 1.0f;

    return (c1 * excess / n) + (c2 * disjoint / n) + (c3 * avg_weight);
}

std::string Genome::to_json() const {
    nlohmann::json j;
    j["num_inputs"] = num_inputs_;
    j["num_outputs"] = num_outputs_;
    j["fitness"] = fitness_;

    j["nodes"] = nlohmann::json::array();
    for (const auto& n : nodes_) {
        j["nodes"].push_back({
            {"id", n.id},
            {"type", static_cast<int>(n.type)}
        });
    }

    j["connections"] = nlohmann::json::array();
    for (const auto& c : connections_) {
        j["connections"].push_back({
            {"in", c.in_node},
            {"out", c.out_node},
            {"weight", c.weight},
            {"enabled", c.enabled},
            {"innovation", c.innovation}
        });
    }

    return j.dump();
}

Genome Genome::from_json(const std::string& json_str) {
    auto j = nlohmann::json::parse(json_str);

    Genome g;
    g.num_inputs_ = j["num_inputs"];
    g.num_outputs_ = j["num_outputs"];
    g.fitness_ = j.value("fitness", 0.0f);

    for (const auto& n : j["nodes"]) {
        g.nodes_.push_back({n["id"], static_cast<NodeType>(n["type"].get<int>())});
    }

    for (const auto& c : j["connections"]) {
        g.connections_.push_back({
            c["in"], c["out"], c["weight"],
            c["enabled"], c["innovation"]
        });
    }

    return g;
}

} // namespace moonai
