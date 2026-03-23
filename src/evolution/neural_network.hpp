#pragma once

#include "evolution/genome.hpp"

#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

namespace moonai {

class NeuralNetwork {
public:
    enum class ActivationFn { Sigmoid, Tanh, ReLU };

    struct Node {
        std::uint32_t id;
        NodeType type;
    };

    explicit NeuralNetwork(const Genome& genome, const std::string& activation_fn = "sigmoid");

    std::vector<float> activate(const std::vector<float>& inputs);
    void activate_into(const float* inputs, int n_in, float* outputs, int n_out);

    int num_nodes() const { return static_cast<int>(nodes_.size()); }
    int num_connections() const { return static_cast<int>(connections_.size()); }
    int num_inputs() const { return num_inputs_; }
    int num_outputs() const { return num_outputs_; }

    // GPU packing accessors — expose precomputed internal structures read-only
    const std::vector<Node>& raw_nodes() const { return nodes_; }
    const std::vector<std::uint32_t>& eval_order() const { return evaluation_order_; }
    const std::vector<std::vector<std::pair<int, float>>>& incoming() const { return incoming_; }
    const std::unordered_map<std::uint32_t, int>& node_index_map() const { return node_index_; }

    // Returns activation values from the most recent activate() call
    const std::vector<float>& last_activations() const { return values_; }

private:

    struct Connection {
        std::uint32_t from;
        std::uint32_t to;
        float weight;
    };

    std::vector<Node> nodes_;
    std::vector<Connection> connections_;
    std::vector<std::uint32_t> evaluation_order_;
    std::unordered_map<std::uint32_t, int> node_index_;  // id -> index in nodes_

    // Precomputed: incoming_[node_idx] = list of {from_node_idx, weight}
    // Built once in build_evaluation_order(), eliminates per-call map allocation.
    std::vector<std::vector<std::pair<int, float>>> incoming_;

    // Reused buffer for node activation values (indexed by node position in nodes_)
    std::vector<float> values_;

    int num_inputs_;
    int num_outputs_;
    ActivationFn activation_fn_ = ActivationFn::Sigmoid;

    void build_evaluation_order();
    static float apply_activation(float x, ActivationFn fn);
};

} // namespace moonai
