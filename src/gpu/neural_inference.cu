#include "gpu/gpu_batch.hpp"
#include "gpu/cuda_utils.cuh"

namespace moonai::gpu {

namespace {

__device__ __forceinline__ float apply_activation(float sum, int activation_fn_id) {
    if (activation_fn_id == 1) {
        return tanhf(sum);
    }
    if (activation_fn_id == 2) {
        return fmaxf(0.0f, sum);
    }
    return 1.0f / (1.0f + expf(-4.9f * sum));
}

} // namespace

// One thread per agent. Reads packed CSR topology, initializes node values
// from d_inputs, runs topological forward pass, writes d_outputs.
__global__ void neural_forward_kernel(
    const GpuNetDesc* __restrict__ descs,
    float* __restrict__            node_vals,
    const uint8_t* __restrict__    node_types,
    const int* __restrict__        eval_order,
    const int* __restrict__        conn_ptr,
    const int* __restrict__        in_count,
    const int* __restrict__        conn_from,
    const float* __restrict__      conn_w,
    const int* __restrict__        out_indices,
    const float* __restrict__      inputs,
    float* __restrict__            outputs,
    int               num_agents,
    int               num_inputs,
    int               activation_fn_id
) {
    int agent_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent_idx >= num_agents) return;

    const GpuNetDesc& desc = descs[agent_idx];

    // ── Initialize node values ───────────────────────────────────────────
    // Input nodes: sequential assignment from d_inputs row for this agent.
    // Bias nodes: 1.0f. Hidden/Output nodes: 0.0f (cleared before forward pass).
    int input_counter = 0;
    for (int i = 0; i < desc.num_nodes; ++i) {
        uint8_t t = node_types[desc.node_off + i];
        if (t == 0) {  // Input
            node_vals[desc.node_off + i] =
                (input_counter < num_inputs)
                    ? inputs[agent_idx * num_inputs + input_counter]
                    : 0.0f;
            ++input_counter;
        } else if (t == 1) {  // Bias
            node_vals[desc.node_off + i] = 1.0f;
        } else {
            node_vals[desc.node_off + i] = 0.0f;
        }
    }

    // ── Forward pass (topological order) ────────────────────────────────
    for (int j = 0; j < desc.num_eval; ++j) {
        int   ni    = eval_order[desc.eval_off + j];
        int   start = conn_ptr  [desc.eval_off + j];
        int   count = in_count  [desc.eval_off + j];

        float sum = 0.0f;
        for (int k = 0; k < count; ++k) {
            int   from_idx = conn_from[desc.conn_off + start + k];
            float w        = conn_w   [desc.conn_off + start + k];
            sum += node_vals[desc.node_off + from_idx] * w;
        }

        node_vals[desc.node_off + ni] = apply_activation(sum, activation_fn_id);
    }

    // ── Write outputs ────────────────────────────────────────────────────
    for (int j = 0; j < desc.num_outputs; ++j) {
        int out_idx = out_indices[desc.out_off + j];
        outputs[agent_idx * desc.num_outputs + j] = node_vals[desc.node_off + out_idx];
    }
}

// Launches kernel on the batch's stream.
void batch_neural_inference(GpuBatch& batch) {
    int n = batch.num_agents();
    int min_grid_size = 0;
    int block_size = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size,
        &block_size,
        neural_forward_kernel,
        0,
        0));
    if (block_size <= 0) {
        block_size = 256;
    }
    int grid_size = (n + block_size - 1) / block_size;

    cudaStream_t stream = static_cast<cudaStream_t>(batch.stream_handle());

    neural_forward_kernel<<<grid_size, block_size, 0, stream>>>(
        batch.d_descs(),
        batch.d_node_vals(),
        batch.d_node_types(),
        batch.d_eval_order(),
        batch.d_conn_ptr(),
        batch.d_in_count(),
        batch.d_conn_from(),
        batch.d_conn_w(),
        batch.d_out_indices(),
        batch.d_inputs(),
        batch.d_outputs(),
        n,
        batch.num_inputs(),
        batch.activation_fn_id()
    );

    CUDA_CHECK(cudaGetLastError());
}

} // namespace moonai::gpu
