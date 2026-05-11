#pragma once

#include "core/types.hpp"
#include "evolution/genome.hpp"

#include <vector>

namespace moonai {

class Species {
public:
  explicit Species(const Genome &representative);

  struct Member {
    uint32_t entity = INVALID_ENTITY;
    int complexity = 0;
  };

  bool is_compatible(const Genome &genome, float threshold, float c1, float c2, float c3,
                     float min_normalization = 1.0f) const;

  void add_member(uint32_t entity, const Genome &genome);
  void clear_members();
  void refresh_summary();
  void set_representative(const Genome &genome) {
    representative_ = genome;
  }

  const Genome &representative() const {
    return representative_;
  }
  const std::vector<Member> &members() const {
    return members_;
  }
  float average_complexity() const {
    return average_complexity_;
  }
  int id() const {
    return id_;
  }

  static int next_species_id();
  static void reset_id_counter();

private:
  int id_;
  Genome representative_;
  std::vector<Member> members_;
  float average_complexity_ = 0.0f;

  static int species_id_counter_;
};

} // namespace moonai
