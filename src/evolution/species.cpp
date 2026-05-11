#include "evolution/species.hpp"

#include <numeric>

namespace moonai {

int Species::species_id_counter_ = 0;

int Species::next_species_id() {
  return species_id_counter_++;
}

void Species::reset_id_counter() {
  species_id_counter_ = 0;
}

Species::Species(const Genome &representative) : id_(next_species_id()), representative_(representative) {}

bool Species::is_compatible(const Genome &genome, float threshold, float c1, float c2, float c3,
                            float min_normalization) const {
  return Genome::compatibility_distance(representative_, genome, c1, c2, c3, min_normalization) < threshold;
}

void Species::add_member(uint32_t entity, const Genome &genome) {
  Member member;
  member.entity = entity;
  member.complexity = genome.complexity();
  members_.push_back(member);
}

void Species::clear_members() {
  members_.clear();
  average_complexity_ = 0.0f;
}

void Species::refresh_summary() {
  if (members_.empty()) {
    average_complexity_ = 0.0f;
    return;
  }

  const float size = static_cast<float>(members_.size());
  const float total_complexity =
      std::accumulate(members_.begin(), members_.end(), 0.0f,
                      [](float sum, const Member &member) { return sum + static_cast<float>(member.complexity); });
  average_complexity_ = total_complexity / size;
}

} // namespace moonai
