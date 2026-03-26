#include "evolution/crossover.hpp"

#include <map>
#include <unordered_set>

namespace moonai {

Genome Crossover::crossover(const Genome &parent_a, const Genome &parent_b,
                            Random &rng) {
  const Genome &fitter =
      (parent_a.fitness() >= parent_b.fitness()) ? parent_a : parent_b;
  const Genome &other = (&fitter == &parent_a) ? parent_b : parent_a;

  Genome child(fitter.num_inputs(), fitter.num_outputs());

  std::map<std::uint32_t, const ConnectionGene *> map_fitter, map_other;
  for (const auto &c : fitter.connections())
    map_fitter[c.innovation] = &c;
  for (const auto &c : other.connections())
    map_other[c.innovation] = &c;

  std::unordered_set<std::uint32_t> needed_nodes;

  for (const auto &[innov, conn] : map_fitter) {
    ConnectionGene gene;

    if (map_other.count(innov)) {
      if (rng.next_bool(0.5f)) {
        gene = *conn;
      } else {
        gene = *map_other.at(innov);
      }

      if (!conn->enabled || !map_other.at(innov)->enabled) {
        gene.enabled = !rng.next_bool(0.75f);
      }
    } else {
      gene = *conn;
    }

    child.add_connection(gene);
    needed_nodes.insert(gene.in_node);
    needed_nodes.insert(gene.out_node);
  }

  for (const auto &node : fitter.nodes()) {
    if (node.type == NodeType::Hidden && needed_nodes.count(node.id)) {
      child.add_node(node);
    }
  }
  for (const auto &node : other.nodes()) {
    if (node.type == NodeType::Hidden && needed_nodes.count(node.id) &&
        !child.has_node(node.id)) {
      child.add_node(node);
    }
  }

  return child;
}

} // namespace moonai
