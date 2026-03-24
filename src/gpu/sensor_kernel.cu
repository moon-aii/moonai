#include "gpu/gpu_batch.hpp"
#include "gpu/cuda_utils.cuh"
#include "gpu/sensor_common.cuh"

namespace moonai::gpu {

namespace {

constexpr int kSensorBlockSize = 256;

template<bool HasWalls>
__global__ void sensor_build_kernel(SensorBuildView view) {
    const int agent_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent_idx >= view.num_agents) {
        return;
    }
    build_sensor_inputs_for_agent<HasWalls>(view, agent_idx);
}

} // namespace

void batch_build_sensors(GpuBatch& batch, float world_width, float world_height,
                         float max_energy, bool has_walls) {
    batch_rebuild_compact_bins(batch);

    const int n = batch.num_agents();
    const int grid_size = (n + kSensorBlockSize - 1) / kSensorBlockSize;
    cudaStream_t stream = static_cast<cudaStream_t>(batch.stream_handle());

    SensorBuildView view{
        batch.d_agent_pos_x(),
        batch.d_agent_pos_y(),
        batch.d_agent_vel_x(),
        batch.d_agent_vel_y(),
        batch.d_agent_speed(),
        batch.d_agent_vision(),
        batch.d_agent_energy(),
        batch.d_agent_ids(),
        batch.d_agent_types(),
        batch.d_agent_alive(),
        batch.d_food_pos_x(),
        batch.d_food_pos_y(),
        batch.d_food_active(),
        batch.d_agent_cell_offsets(),
        batch.d_agent_cell_ids(),
        batch.d_food_cell_offsets(),
        batch.d_food_cell_ids(),
        batch.d_inputs(),
        n,
        batch.food_count(),
        batch.agent_cols(),
        batch.agent_rows(),
        batch.agent_cell_size(),
        batch.food_cols(),
        batch.food_rows(),
        batch.food_cell_size(),
        batch.num_inputs(),
        world_width,
        world_height,
        max_energy,
        has_walls,
    };

    if (has_walls) {
        sensor_build_kernel<true><<<grid_size, kSensorBlockSize, 0, stream>>>(view);
    } else {
        sensor_build_kernel<false><<<grid_size, kSensorBlockSize, 0, stream>>>(view);
    }

    CUDA_CHECK(cudaGetLastError());
}

} // namespace moonai::gpu
