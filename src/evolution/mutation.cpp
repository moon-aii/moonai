#include "evolution/mutation.hpp"

#include <algorithm>
#include <stack>
#include <unordered_set>

namespace moonai {

void InnovationTracker::init_from_population(const std::vector<Genome> &population) {
  innovation_counter_ = 0;
  node_counter_ = 0;
  innovation_cache_.clear();
  split_node_cache_.clear();

  for (const auto &genome : population) {
    for (const auto &conn : genome.connections()) {
      innovation_cache_[std::make_pair(conn.in_node, conn.out_node)] = conn.innovation;
      if (conn.innovation >= innovation_counter_) {
        innovation_counter_ = conn.innovation + 1;
      }
    }
    for (const auto &node : genome.nodes()) {
      if (node.id >= node_counter_) {
        node_counter_ = node.id + 1;
      }
    }
  }
}

std::uint32_t InnovationTracker::get_innovation(std::uint32_t in_node, std::uint32_t out_node) {
  auto key = std::make_pair(in_node, out_node);
  auto it = innovation_cache_.find(key);
  if (it != innovation_cache_.end()) {
    return it->second;
  }
  std::uint32_t innov = innovation_counter_++;
  innovation_cache_[key] = innov;
  return innov;
}

std::uint32_t InnovationTracker::next_node_id() {
  return node_counter_++;
}

std::uint32_t InnovationTracker::get_split_node_id(std::uint32_t in_node, std::uint32_t out_node) {
  const auto key = std::make_pair(in_node, out_node);
  const auto it = split_node_cache_.find(key);
  if (it != split_node_cache_.end()) {
    return it->second;
  }

  const std::uint32_t node_id = next_node_id();
  split_node_cache_[key] = node_id;
  return node_id;
}

void Mutation::mutate_weights(Genome &genome, Random &rng, float power) {
  for (auto &conn : genome.connections()) {
    if (rng.next_bool(0.9f)) {
      conn.weight += rng.next_gaussian(0.0f, power);
      conn.weight = std::clamp(conn.weight, -8.0f, 8.0f);
    } else {
      conn.weight = rng.next_float(-2.0f, 2.0f);
    }
  }
}

void Mutation::add_connection(Genome &genome, Random &rng, InnovationTracker &tracker) {
  const auto &nodes = genome.nodes();
  if (nodes.size() < 2)
    return;

  auto would_create_cycle = [&](std::uint32_t from_id, std::uint32_t to_id) -> bool {
    std::unordered_set<std::uint32_t> visited;
    std::stack<std::uint32_t> stack;
    stack.push(to_id);
    while (!stack.empty()) {
      auto nid = stack.top();
      stack.pop();
      if (nid == from_id)
        return true;
      if (!visited.insert(nid).second)
        continue;
      for (const auto &c : genome.connections()) {
        if (c.enabled && c.in_node == nid)
          stack.push(c.out_node);
      }
    }
    return false;
  };

  for (int attempt = 0; attempt < 30; ++attempt) {
    int from_idx = rng.next_int(0, static_cast<int>(nodes.size()) - 1);
    int to_idx = rng.next_int(0, static_cast<int>(nodes.size()) - 1);

    const auto &from = nodes[from_idx];
    const auto &to = nodes[to_idx];

    if (to.type == NodeType::Input || to.type == NodeType::Bias)
      continue;
    if (from.type == NodeType::Output)
      continue;
    if (from.id == to.id)
      continue;

    if (genome.has_connection(from.id, to.id))
      continue;

    if (would_create_cycle(from.id, to.id))
      continue;

    std::uint32_t innov = tracker.get_innovation(from.id, to.id);
    genome.add_connection({from.id, to.id, rng.next_float(-1.0f, 1.0f), true, innov});
    return;
  }
}

void Mutation::add_node(Genome &genome, Random &rng, InnovationTracker &tracker, int max_hidden_nodes) {
  if (max_hidden_nodes > 0) {
    int hidden_count = static_cast<int>(std::count_if(genome.nodes().begin(), genome.nodes().end(),
                                                      [](const auto &n) { return n.type == NodeType::Hidden; }));
    if (hidden_count >= max_hidden_nodes)
      return;
  }

  auto &conns = genome.connections();
  if (conns.empty())
    return;

  std::vector<int> indices(conns.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::vector<int> enabled_indices;
  enabled_indices.reserve(conns.size());
  std::copy_if(indices.begin(), indices.end(), std::back_inserter(enabled_indices),
               [&conns](int i) { return conns[i].enabled; });
  if (enabled_indices.empty())
    return;

  int idx = enabled_indices[rng.next_int(0, static_cast<int>(enabled_indices.size()) - 1)];

  std::uint32_t in_id = conns[idx].in_node;
  std::uint32_t out_id = conns[idx].out_node;
  float old_weight = conns[idx].weight;
  conns[idx].enabled = false;

  const std::uint32_t new_id = tracker.get_split_node_id(in_id, out_id);
  if (!genome.has_node(new_id)) {
    genome.add_node({new_id, NodeType::Hidden});
  }

  auto ensure_connection = [&](std::uint32_t from_id, std::uint32_t to_id, float weight, std::uint32_t innovation) {
    for (auto &conn : conns) {
      if (conn.in_node == from_id && conn.out_node == to_id) {
        conn.enabled = true;
        return;
      }
    }

    genome.add_connection({from_id, to_id, weight, true, innovation});
  };

  const std::uint32_t innov1 = tracker.get_innovation(in_id, new_id);
  ensure_connection(in_id, new_id, 1.0f, innov1);

  const std::uint32_t innov2 = tracker.get_innovation(new_id, out_id);
  ensure_connection(new_id, out_id, old_weight, innov2);
}

void Mutation::delete_connection(Genome &genome, Random &rng) {
  auto &conns = genome.connections();
  if (conns.size() <= 1)
    return; // Keep at least one connection

  int idx = rng.next_int(0, static_cast<int>(conns.size()) - 1);
  conns.erase(conns.begin() + idx);
}

void Mutation::mutate(Genome &genome, Random &rng, const SimulationConfig &config, InnovationTracker &tracker) {
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

  bool any_enabled = std::any_of(genome.connections().begin(), genome.connections().end(),
                                 [](const auto &conn) { return conn.enabled; });
  if (!any_enabled && !genome.connections().empty()) {
    int idx = rng.next_int(0, static_cast<int>(genome.connections().size()) - 1);
    genome.connections()[idx].enabled = true;
  }
}

} // namespace moonai
