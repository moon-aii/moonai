#include "evolution/neural_network.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <queue>
#include <unordered_set>

namespace moonai {

NeuralNetwork::NeuralNetwork(const Genome &genome)
    : num_inputs_(genome.num_inputs()), num_outputs_(genome.num_outputs()) {
  for (const auto &ng : genome.nodes()) {
    node_index_[ng.id] = static_cast<int>(nodes_.size());
    nodes_.push_back({ng.id, ng.type});
  }

  for (const auto &cg : genome.connections()) {
    if (cg.enabled) {
      connections_.push_back({cg.in_node, cg.out_node, cg.weight});
    }
  }

  values_.assign(nodes_.size(), 0.0f);
  build_evaluation_order();
}

std::vector<float> NeuralNetwork::activate(const std::vector<float> &inputs) {
  std::fill(values_.begin(), values_.end(), 0.0f);

  int idx = 0;
  for (size_t i = 0; i < nodes_.size(); ++i) {
    if (nodes_[i].type == NodeType::Input && idx < static_cast<int>(inputs.size())) {
      values_[i] = inputs[idx++];
    } else if (nodes_[i].type == NodeType::Bias) {
      values_[i] = 1.0f;
    }
  }

  for (auto node_id : evaluation_order_) {
    int ni = node_index_.at(node_id);
    float sum = 0.0f;
    for (const auto &[from_idx, w] : incoming_[ni]) {
      sum += values_[from_idx] * w;
    }
    values_[ni] = apply_activation(sum);
  }

  std::vector<float> outputs;
  for (size_t i = 0; i < nodes_.size(); ++i) {
    if (nodes_[i].type == NodeType::Output) {
      outputs.push_back(values_[i]);
    }
  }

  return outputs;
}

void NeuralNetwork::activate_into(const float *inputs, int n_in, float *outputs, int n_out) {
  std::fill(values_.begin(), values_.end(), 0.0f);

  int idx = 0;
  for (size_t i = 0; i < nodes_.size(); ++i) {
    if (nodes_[i].type == NodeType::Input && idx < n_in) {
      values_[i] = inputs[idx++];
    } else if (nodes_[i].type == NodeType::Bias) {
      values_[i] = 1.0f;
    }
  }

  for (auto node_id : evaluation_order_) {
    int ni = node_index_.at(node_id);
    float sum = 0.0f;
    for (const auto &[from_idx, w] : incoming_[ni]) {
      sum += values_[from_idx] * w;
    }
    values_[ni] = apply_activation(sum);
  }

  int out_idx = 0;
  for (size_t i = 0; i < nodes_.size() && out_idx < n_out; ++i) {
    if (nodes_[i].type == NodeType::Output) {
      outputs[out_idx++] = values_[i];
    }
  }
}

void NeuralNetwork::build_evaluation_order() {
  evaluation_order_.clear();

  std::unordered_set<std::uint32_t> eval_nodes;
  for (const auto &node : nodes_) {
    if (node.type == NodeType::Hidden || node.type == NodeType::Output) {
      eval_nodes.insert(node.id);
    }
  }

  std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> adj;
  std::unordered_map<std::uint32_t, int> in_degree;

  for (auto nid : eval_nodes) {
    in_degree[nid] = 0;
  }

  for (const auto &conn : connections_) {
    if (eval_nodes.count(conn.to)) {
      adj[conn.from].push_back(conn.to);
      if (eval_nodes.count(conn.from)) {
        in_degree[conn.to]++;
      }
    }
  }

  std::queue<std::uint32_t> ready;
  for (auto nid : eval_nodes) {
    if (in_degree[nid] == 0) {
      ready.push(nid);
    }
  }

  while (!ready.empty()) {
    auto nid = ready.front();
    ready.pop();
    evaluation_order_.push_back(nid);

    for (auto to_id : adj[nid]) {
      if (eval_nodes.count(to_id)) {
        in_degree[to_id]--;
        if (in_degree[to_id] == 0) {
          ready.push(to_id);
        }
      }
    }
  }

  std::unordered_set<std::uint32_t> ordered_set(evaluation_order_.begin(), evaluation_order_.end());
  std::vector<std::uint32_t> cycle_nodes;
  std::copy_if(eval_nodes.begin(), eval_nodes.end(), std::back_inserter(cycle_nodes),
               [&ordered_set](auto nid) { return !ordered_set.count(nid); });
  std::sort(cycle_nodes.begin(), cycle_nodes.end());
  evaluation_order_.insert(evaluation_order_.end(), cycle_nodes.begin(), cycle_nodes.end());

  incoming_.assign(nodes_.size(), {});
  for (const auto &conn : connections_) {
    auto it_to = node_index_.find(conn.to);
    auto it_from = node_index_.find(conn.from);
    if (it_to != node_index_.end() && it_from != node_index_.end()) {
      incoming_[it_to->second].emplace_back(it_from->second, conn.weight);
    }
  }
}

float NeuralNetwork::apply_activation(float x) {
  return std::tanh(x);
}

int NeuralNetwork::num_input_nodes() const {
  int count = 0;
  for (const auto &node : nodes_) {
    if (node.type == NodeType::Input) {
      count++;
    }
  }
  return count;
}

int NeuralNetwork::num_output_nodes() const {
  int count = 0;
  for (const auto &node : nodes_) {
    if (node.type == NodeType::Output) {
      count++;
    }
  }
  return count;
}

std::vector<NeuralNetwork::IncomingConnection> NeuralNetwork::get_incoming_connections(int node_idx) const {
  std::vector<IncomingConnection> result;
  if (node_idx >= 0 && node_idx < static_cast<int>(incoming_.size())) {
    for (const auto &[from_idx, weight] : incoming_[node_idx]) {
      result.push_back({from_idx, weight});
    }
  }
  return result;
}

std::vector<int> NeuralNetwork::get_output_indices() const {
  std::vector<int> indices;
  for (size_t i = 0; i < nodes_.size(); ++i) {
    if (nodes_[i].type == NodeType::Output) {
      indices.push_back(static_cast<int>(i));
    }
  }
  return indices;
}

} // namespace moonai
