#include "gpu/gpu_batch.hpp"
#include "gpu/cuda_utils.cuh"

namespace moonai::gpu {

// One thread per agent. Reads packed CSR topology, initializes node values
// from d_inputs, runs topological forward pass, writes d_outputs.
__global__ void neural_forward_kernel(
    const GpuNetDesc* descs,
    float*            node_vals,
    const uint8_t*    node_types,
    const int*        eval_order,
    const int*        conn_ptr,
    const int*        in_count,
    const int*        conn_from,
    const float*      conn_w,
    const int*        out_indices,
    const float*      inputs,
    float*            outputs,
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

        float act;
        if (activation_fn_id == 1) {          // tanh
            act = tanhf(sum);
        } else if (activation_fn_id == 2) {   // relu
            act = fmaxf(0.0f, sum);
        } else {                               // sigmoid (default)
            act = 1.0f / (1.0f + expf(-4.9f * sum));
        }
        node_vals[desc.node_off + ni] = act;
    }

    // ── Write outputs ────────────────────────────────────────────────────
    for (int j = 0; j < desc.num_outputs; ++j) {
        int out_idx = out_indices[desc.out_off + j];
        outputs[agent_idx * desc.num_outputs + j] = node_vals[desc.node_off + out_idx];
    }
}

// Synchronous: launches on default stream and waits for completion.
void batch_neural_inference(GpuBatch& batch) {
    int n = batch.num_agents();
    constexpr int kBlockSize = 256;
    int grid_size = (n + kBlockSize - 1) / kBlockSize;

    neural_forward_kernel<<<grid_size, kBlockSize>>>(
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
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Async: launches on stream[buf] using double-buffered device arrays.
// Caller is responsible for synchronization via GpuBatch::sync_stream().
void batch_neural_inference(GpuBatch& batch, int buf) {
    int n = batch.num_agents();
    constexpr int kBlockSize = 256;
    int grid_size = (n + kBlockSize - 1) / kBlockSize;

    cudaStream_t stream = static_cast<cudaStream_t>(batch.stream_handle(buf));

    neural_forward_kernel<<<grid_size, kBlockSize, 0, stream>>>(
        batch.d_descs(),
        batch.d_node_vals(),
        batch.d_node_types(),
        batch.d_eval_order(),
        batch.d_conn_ptr(),
        batch.d_in_count(),
        batch.d_conn_from(),
        batch.d_conn_w(),
        batch.d_out_indices(),
        batch.d_inputs_buf(buf),
        batch.d_outputs_buf(buf),
        n,
        batch.num_inputs(),
        batch.activation_fn_id()
    );

    CUDA_CHECK(cudaGetLastError());
}

} // namespace moonai::gpu
