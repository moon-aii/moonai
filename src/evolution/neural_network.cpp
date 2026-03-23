#include "evolution/neural_network.hpp"

#include <cmath>
#include <algorithm>
#include <queue>
#include <unordered_set>

namespace moonai {

NeuralNetwork::NeuralNetwork(const Genome& genome, const std::string& activation_fn)
    : num_inputs_(genome.num_inputs())
    , num_outputs_(genome.num_outputs()) {
    if (activation_fn == "tanh")
        activation_fn_ = ActivationFn::Tanh;
    else if (activation_fn == "relu")
        activation_fn_ = ActivationFn::ReLU;
    else
        activation_fn_ = ActivationFn::Sigmoid;
    for (const auto& ng : genome.nodes()) {
        node_index_[ng.id] = static_cast<int>(nodes_.size());
        nodes_.push_back({ng.id, ng.type});
    }

    for (const auto& cg : genome.connections()) {
        if (cg.enabled) {
            connections_.push_back({cg.in_node, cg.out_node, cg.weight});
        }
    }

    values_.assign(nodes_.size(), 0.0f);
    build_evaluation_order();
}

std::vector<float> NeuralNetwork::activate(const std::vector<float>& inputs) {
    // Reset all node values
    std::fill(values_.begin(), values_.end(), 0.0f);

    // Set input and bias values
    int idx = 0;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i].type == NodeType::Input && idx < static_cast<int>(inputs.size())) {
            values_[i] = inputs[idx++];
        } else if (nodes_[i].type == NodeType::Bias) {
            values_[i] = 1.0f;
        }
    }

    // Evaluate in topological order using precomputed incoming adjacency list
    for (auto node_id : evaluation_order_) {
        int ni = node_index_.at(node_id);
        float sum = 0.0f;
        for (const auto& [from_idx, w] : incoming_[ni]) {
            sum += values_[from_idx] * w;
        }
        values_[ni] = apply_activation(sum, activation_fn_);
    }

    // Collect outputs in node order
    std::vector<float> outputs;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i].type == NodeType::Output) {
            outputs.push_back(values_[i]);
        }
    }

    return outputs;
}

void NeuralNetwork::activate_into(const float* inputs, int n_in, float* outputs, int n_out) {
    // Reset all node values
    std::fill(values_.begin(), values_.end(), 0.0f);

    // Set input and bias values
    int idx = 0;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i].type == NodeType::Input && idx < n_in) {
            values_[i] = inputs[idx++];
        } else if (nodes_[i].type == NodeType::Bias) {
            values_[i] = 1.0f;
        }
    }

    // Evaluate in topological order using precomputed incoming adjacency list
    for (auto node_id : evaluation_order_) {
        int ni = node_index_.at(node_id);
        float sum = 0.0f;
        for (const auto& [from_idx, w] : incoming_[ni]) {
            sum += values_[from_idx] * w;
        }
        values_[ni] = apply_activation(sum, activation_fn_);
    }

    // Collect outputs in node order
    int out_idx = 0;
    for (size_t i = 0; i < nodes_.size() && out_idx < n_out; ++i) {
        if (nodes_[i].type == NodeType::Output) {
            outputs[out_idx++] = values_[i];
        }
    }
}

void NeuralNetwork::build_evaluation_order() {
    // Kahn's algorithm for topological sort
    // Only sort hidden and output nodes (inputs/bias don't need evaluation)
    evaluation_order_.clear();

    // Collect nodes that need evaluation
    std::unordered_set<std::uint32_t> eval_nodes;
    for (const auto& node : nodes_) {
        if (node.type == NodeType::Hidden || node.type == NodeType::Output) {
            eval_nodes.insert(node.id);
        }
    }

    // Build adjacency list and in-degree count (only for eval nodes)
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> adj;  // from -> [to]
    std::unordered_map<std::uint32_t, int> in_degree;

    for (auto nid : eval_nodes) {
        in_degree[nid] = 0;
    }

    for (const auto& conn : connections_) {
        if (eval_nodes.count(conn.to)) {
            adj[conn.from].push_back(conn.to);
            // Only count in-degree from other eval nodes.
            // Connections from input/bias nodes don't block evaluation
            // because input values are pre-set before the eval loop.
            if (eval_nodes.count(conn.from)) {
                in_degree[conn.to]++;
            }
        }
    }

    // Start with nodes that have no incoming connections from other eval nodes
    // (or only from input/bias nodes)
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

    // If some nodes weren't reached (cycles), append them sorted for determinism
    // This handles recurrent connections gracefully
    std::vector<std::uint32_t> cycle_nodes;
    std::unordered_set<std::uint32_t> ordered_set(
        evaluation_order_.begin(), evaluation_order_.end());
    for (auto nid : eval_nodes) {
        if (!ordered_set.count(nid)) {
            cycle_nodes.push_back(nid);
        }
    }
    std::sort(cycle_nodes.begin(), cycle_nodes.end());
    for (auto nid : cycle_nodes) {
        evaluation_order_.push_back(nid);
    }

    // Build incoming adjacency list: incoming_[node_idx] = {from_idx, weight}
    // This is used by activate() to avoid per-call map allocation.
    incoming_.assign(nodes_.size(), {});
    for (const auto& conn : connections_) {
        auto it_to   = node_index_.find(conn.to);
        auto it_from = node_index_.find(conn.from);
        if (it_to != node_index_.end() && it_from != node_index_.end()) {
            incoming_[it_to->second].emplace_back(it_from->second, conn.weight);
        }
    }
}

float NeuralNetwork::apply_activation(float x, ActivationFn fn) {
    switch (fn) {
        case ActivationFn::Tanh: return std::tanh(x);
        case ActivationFn::ReLU: return std::max(0.0f, x);
        default:                 return 1.0f / (1.0f + std::exp(-4.9f * x));
    }
}

} // namespace moonai
