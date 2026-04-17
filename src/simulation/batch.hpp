#pragma once

#include "simulation/buffers.hpp"
#include "simulation/layout.hpp"

#include <cstddef>

#ifndef __CUDACC__
typedef struct CUstream_st *cudaStream_t;
#endif

namespace moonai::simulation {

struct StepParams {
  float world_width = 0.0f;
  float world_height = 0.0f;
  float energy_drain_per_step = 0.0f;
  float vision_range = 120.0f;
  float max_energy = 150.0f;
  int max_age = 0;
  float interaction_range = 12.0f;
  float energy_gain_from_food = 40.0f;
  float energy_gain_from_kill = 60.0f;
  float predator_speed = 0.6f;
  float prey_speed = 0.66f;
};

class Batch {
public:
  Batch();
  Batch(std::size_t max_predators, std::size_t max_prey, std::size_t max_food);
  ~Batch();

  Batch(const Batch &) = delete;
  Batch &operator=(const Batch &) = delete;
  Batch(Batch &&) = delete;
  Batch &operator=(Batch &&) = delete;

  [[nodiscard]] PopulationBuffer &predator_buffer() {
    return predator_buffer_;
  }
  [[nodiscard]] const PopulationBuffer &predator_buffer() const {
    return predator_buffer_;
  }
  [[nodiscard]] PopulationBuffer &prey_buffer() {
    return prey_buffer_;
  }
  [[nodiscard]] const PopulationBuffer &prey_buffer() const {
    return prey_buffer_;
  }
  [[nodiscard]] FoodBuffer &food_buffer() {
    return food_buffer_;
  }
  [[nodiscard]] const FoodBuffer &food_buffer() const {
    return food_buffer_;
  }

  void launch_build_sensors_async(const StepParams &params, std::size_t predator_count, std::size_t prey_count,
                                  std::size_t food_count);
  void launch_post_inference_async(const StepParams &params, std::size_t predator_count, std::size_t prey_count,
                                   std::size_t food_count);

  void upload_async(std::size_t predator_count, std::size_t prey_count, std::size_t food_count);
  void download_async(std::size_t predator_count, std::size_t prey_count, std::size_t food_count);
  void synchronize();
  void mark_error();
  void ensure_capacity(std::size_t predator_count, std::size_t prey_count, std::size_t food_count);

  [[nodiscard]] bool ok() const noexcept {
    return !had_error_;
  }
  [[nodiscard]] cudaStream_t stream() const {
    return static_cast<cudaStream_t>(stream_);
  }

  [[nodiscard]] std::size_t predator_capacity() const noexcept {
    return predator_buffer_.capacity();
  }
  [[nodiscard]] std::size_t prey_capacity() const noexcept {
    return prey_buffer_.capacity();
  }
  [[nodiscard]] std::size_t food_capacity() const noexcept {
    return food_buffer_.capacity();
  }

private:
  PopulationBuffer predator_buffer_;
  PopulationBuffer prey_buffer_;
  FoodBuffer food_buffer_;

  int *d_predator_cell_counts_ = nullptr;
  int *d_predator_cell_offsets_ = nullptr;
  int *d_predator_cell_write_offsets_ = nullptr;
  PopulationEntry *d_predator_grid_entries_ = nullptr;

  int *d_prey_cell_counts_ = nullptr;
  int *d_prey_cell_offsets_ = nullptr;
  int *d_prey_cell_write_offsets_ = nullptr;
  PopulationEntry *d_prey_grid_entries_ = nullptr;

  int *d_food_cell_counts_ = nullptr;
  int *d_food_cell_offsets_ = nullptr;
  int *d_food_cell_write_offsets_ = nullptr;
  FoodEntry *d_food_grid_entries_ = nullptr;

  std::size_t grid_cell_capacity_ = 0;
  int grid_cols_ = 0;
  int grid_rows_ = 0;
  float grid_cell_size_ = 0.0f;

  void *stream_ = nullptr;
  bool had_error_ = false;

  void init_cuda_resources();
  void cleanup_cuda_resources();
  void ensure_spatial_grid_capacity(std::size_t cell_count);
  void free_spatial_grid_buffers();
  void check_launch_error();
};

} // namespace moonai::simulation
