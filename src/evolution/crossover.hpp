#pragma once

#include "core/random.hpp"
#include "evolution/genome.hpp"

namespace moonai {

class Crossover {
public:
  static Genome crossover(const Genome &parent_a, const Genome &parent_b, Random &rng);
};

} // namespace moonai
