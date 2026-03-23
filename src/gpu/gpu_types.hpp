#pragma once

namespace moonai::gpu {

// Per-agent network descriptor for CSR-packed flat GPU layout
struct GpuNetDesc {
    int num_nodes;   // total node count (input+bias+hidden+output)
    int num_eval;    // evaluation order length (hidden+output only)
    int num_inputs;  // number of Input type nodes (excluding bias)
    int num_outputs; // number of output nodes
    int node_off;    // offset into d_node_vals[], d_node_types[]
    int eval_off;    // offset into d_eval_order[], d_conn_ptr[], d_in_count[]
    int conn_off;    // offset into d_conn_from[], d_conn_w[]
    int out_off;     // offset into d_out_indices[]
};

} // namespace moonai::gpu
