#pragma once

#include "core/types.hpp"

namespace moonai {

enum class AgentType { Predator, Prey };

class Agent {
public:
    Agent(AgentId id, AgentType type, Vec2 position, float speed,
          float vision_range, float energy);
    virtual ~Agent() = default;

    virtual void update(float dt) = 0;

    // Apply a movement decision from the neural network (dx, dy normalized)
    void apply_movement(Vec2 direction, float dt);

    AgentId id() const { return id_; }
    AgentType type() const { return type_; }
    Vec2 position() const { return position_; }
    Vec2 velocity() const { return velocity_; }
    float speed() const { return speed_; }
    float vision_range() const { return vision_range_; }
    float energy() const { return energy_; }
    int age() const { return age_; }
    bool alive() const { return alive_; }
    float fitness() const { return fitness_; }
    int kills() const { return kills_; }
    int food_eaten() const { return food_eaten_; }
    float distance_traveled() const { return distance_traveled_; }
    int species_id() const { return species_id_; }

    void set_position(Vec2 pos) { position_ = pos; }
    void set_velocity(Vec2 vel) { velocity_ = vel; }
    void set_energy(float energy) { energy_ = energy; }
    void set_age(int age) { age_ = age; }
    void set_distance_traveled(float distance) { distance_traveled_ = distance; }
    void set_kills(int kills) { kills_ = kills; }
    void set_food_eaten(int food_eaten) { food_eaten_ = food_eaten; }
    void set_species_id(int id) { species_id_ = id; }
    void set_alive(bool alive) { alive_ = alive; }
    void add_fitness(float amount) { fitness_ += amount; }
    void add_energy(float amount) { energy_ += amount; }
    void drain_energy(float amount) { energy_ -= amount; }
    void increment_age() { ++age_; }
    void add_kill() { ++kills_; }
    void add_food() { ++food_eaten_; }

    bool is_dead() const { return energy_ <= 0.0f; }

protected:
    AgentId id_;
    AgentType type_;
    Vec2 position_;
    Vec2 velocity_ = {0.0f, 0.0f};
    float speed_;
    float vision_range_;
    float energy_;
    int age_ = 0;
    bool alive_ = true;
    float fitness_ = 0.0f;
    int kills_ = 0;
    int food_eaten_ = 0;
    float distance_traveled_ = 0.0f;
    int species_id_ = -1;
};

} // namespace moonai
