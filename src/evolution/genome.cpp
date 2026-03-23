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

float Genome::compatibility_distance(const Genome& a, const Genome& b,
                                     float c1, float c2, float c3) {
    ScopedTimer timer(ProfileEvent::CompatibilityDistance);
    Profiler::instance().increment(ProfileCounter::CompatibilityChecks);
    const auto& conns_a = a.connections();
    const auto& conns_b = b.connections();

    if (conns_a.empty() && conns_b.empty()) return 0.0f;

    std::map<std::uint32_t, const ConnectionGene*> map_a, map_b;
    for (const auto& c : conns_a) map_a[c.innovation] = &c;
    for (const auto& c : conns_b) map_b[c.innovation] = &c;

    int excess = 0, disjoint = 0, matching = 0;
    float weight_diff = 0.0f;

    std::uint32_t max_a = 0;
    for (const auto& c : conns_a) if (c.innovation > max_a) max_a = c.innovation;
    std::uint32_t max_b = 0;
    for (const auto& c : conns_b) if (c.innovation > max_b) max_b = c.innovation;

    std::uint32_t max_innov = std::max(max_a, max_b);
    std::uint32_t min_max = std::min(max_a, max_b);

    for (std::uint32_t i = 0; i <= max_innov; ++i) {
        bool in_a = map_a.count(i) > 0;
        bool in_b = map_b.count(i) > 0;

        if (in_a && in_b) {
            ++matching;
            weight_diff += std::abs(map_a[i]->weight - map_b[i]->weight);
        } else if (in_a || in_b) {
            if (i > min_max) {
                ++excess;
            } else {
                ++disjoint;
            }
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
