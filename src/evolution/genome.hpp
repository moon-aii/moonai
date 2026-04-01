#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace moonai {

enum class NodeType { Input, Hidden, Output, Bias };

struct NodeGene {
  std::uint32_t id;
  NodeType type;
};

struct ConnectionGene {
  std::uint32_t in_node;
  std::uint32_t out_node;
  float weight;
  bool enabled;
  std::uint32_t innovation;
};

class Genome {
public:
  Genome() = default;
  Genome(int num_inputs, int num_outputs);

  void add_node(const NodeGene &node);
  void add_connection(const ConnectionGene &conn);

  bool has_connection(std::uint32_t from, std::uint32_t to) const;
  bool has_node(std::uint32_t id) const;
  std::uint32_t max_node_id() const;

  const std::vector<NodeGene> &nodes() const {
    return nodes_;
  }
  const std::vector<ConnectionGene> &connections() const {
    return connections_;
  }
  std::vector<ConnectionGene> &connections() {
    return connections_;
  }

  int num_inputs() const {
    return num_inputs_;
  }
  int num_outputs() const {
    return num_outputs_;
  }

  // Genome complexity (nodes + connections)
  int complexity() const;

  static float compatibility_distance(const Genome &a, const Genome &b, float c1, float c2, float c3);

  // JSON serialization
  std::string to_json() const;
  static Genome from_json(const std::string &json_str);

private:
  std::vector<NodeGene> nodes_;
  std::vector<ConnectionGene> connections_;
  int num_inputs_ = 0;
  int num_outputs_ = 0;
};

} // namespace moonai
