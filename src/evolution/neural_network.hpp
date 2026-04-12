#pragma once

#include "evolution/genome.hpp"

#include <unordered_map>
#include <utility>
#include <vector>

namespace moonai {

class NeuralNetwork {
public:
  struct Node {
    std::uint32_t id;
    NodeType type;
  };

  explicit NeuralNetwork(const Genome &genome);

  std::vector<float> activate(const std::vector<float> &inputs);
  void activate_into(const float *inputs, int n_in, float *outputs, int n_out);

  int num_nodes() const {
    return static_cast<int>(nodes_.size());
  }
  int num_connections() const {
    return static_cast<int>(connections_.size());
  }
  int num_inputs() const {
    return num_inputs_;
  }
  int num_outputs() const {
    return num_outputs_;
  }

  const std::vector<Node> &raw_nodes() const {
    return nodes_;
  }
  const std::vector<std::uint32_t> &eval_order() const {
    return evaluation_order_;
  }
  const std::vector<std::vector<std::pair<int, float>>> &incoming() const {
    return incoming_;
  }
  const std::unordered_map<std::uint32_t, int> &node_index_map() const {
    return node_index_;
  }

  // Returns activation values from the most recent activate() call
  const std::vector<float> &last_activations() const {
    return values_;
  }

  int num_input_nodes() const;
  int num_output_nodes() const;

  struct IncomingConnection {
    int from_node;
    float weight;
  };
  std::vector<IncomingConnection> get_incoming_connections(int node_idx) const;
  std::vector<int> get_output_indices() const;

private:
  struct Connection {
    std::uint32_t from;
    std::uint32_t to;
    float weight;
  };

  std::vector<Node> nodes_;
  std::vector<Connection> connections_;
  std::vector<std::uint32_t> evaluation_order_;
  std::unordered_map<std::uint32_t, int> node_index_;

  std::vector<std::vector<std::pair<int, float>>> incoming_;

  std::vector<float> values_;

  int num_inputs_;
  int num_outputs_;

  void build_evaluation_order();
  static float apply_activation(float x);
};

} // namespace moonai
