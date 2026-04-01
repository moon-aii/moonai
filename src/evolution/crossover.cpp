#include "evolution/crossover.hpp"

#include <map>
#include <unordered_set>

namespace moonai {

Genome Crossover::crossover(const Genome &parent_a, const Genome &parent_b, Random &rng) {
  Genome child(parent_a.num_inputs(), parent_a.num_outputs());

  std::map<std::uint32_t, const ConnectionGene *> map_a, map_b;
  for (const auto &c : parent_a.connections())
    map_a[c.innovation] = &c;
  for (const auto &c : parent_b.connections())
    map_b[c.innovation] = &c;

  std::unordered_set<std::uint32_t> needed_nodes;
  std::vector<ConnectionGene> inherited_connections;

  auto inherit_gene = [&](const ConnectionGene &gene) {
    inherited_connections.push_back(gene);
    needed_nodes.insert(gene.in_node);
    needed_nodes.insert(gene.out_node);
  };

  std::unordered_set<std::uint32_t> all_innovations;
  for (const auto &[innov, _] : map_a)
    all_innovations.insert(innov);
  for (const auto &[innov, _] : map_b)
    all_innovations.insert(innov);

  for (std::uint32_t innov : all_innovations) {
    const auto it_a = map_a.find(innov);
    const auto it_b = map_b.find(innov);

    if (it_a != map_a.end() && it_b != map_b.end()) {
      ConnectionGene gene = rng.next_bool(0.5f) ? *it_a->second : *it_b->second;
      if (!it_a->second->enabled || !it_b->second->enabled) {
        gene.enabled = !rng.next_bool(0.75f);
      }
      inherit_gene(gene);
      continue;
    }

    if (!rng.next_bool(0.5f)) {
      continue;
    }

    const ConnectionGene &gene = (it_a != map_a.end()) ? *it_a->second : *it_b->second;
    inherit_gene(gene);
  }

  if (inherited_connections.empty() && !all_innovations.empty()) {
    auto innov_it = all_innovations.begin();
    std::advance(innov_it, rng.next_int(0, static_cast<int>(all_innovations.size()) - 1));
    const auto it_a = map_a.find(*innov_it);
    const ConnectionGene &gene = (it_a != map_a.end()) ? *it_a->second : *map_b.at(*innov_it);
    inherit_gene(gene);
  }

  for (const auto &gene : inherited_connections)
    child.add_connection(gene);

  for (const auto &node : parent_a.nodes()) {
    if (node.type == NodeType::Hidden && needed_nodes.count(node.id)) {
      child.add_node(node);
    }
  }
  for (const auto &node : parent_b.nodes()) {
    if (node.type == NodeType::Hidden && needed_nodes.count(node.id) && !child.has_node(node.id)) {
      child.add_node(node);
    }
  }

  return child;
}

} // namespace moonai
