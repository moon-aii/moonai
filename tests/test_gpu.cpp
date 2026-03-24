#include <gtest/gtest.h>
#include <cmath>
#include "core/config.hpp"
#include "core/random.hpp"
#include "evolution/evolution_manager.hpp"
#include "evolution/genome.hpp"
#include "evolution/mutation.hpp"
#include "evolution/neural_network.hpp"
#include "gpu/gpu_batch.hpp"
#include "simulation/simulation_manager.hpp"

using namespace moonai;
using namespace moonai::gpu;

// ── Helper: build GpuNetworkData from a vector of NeuralNetworks ────────────
// Mirrors the static build_gpu_network_data() in evolution_manager.cpp.
static GpuNetworkData build_test_network_data(
    const std::vector<std::unique_ptr<NeuralNetwork>>& nets,
    const std::string& activation_fn)
{
    GpuNetworkData data;

    if (activation_fn == "tanh")       data.activation_fn_id = 1;
    else if (activation_fn == "relu")  data.activation_fn_id = 2;
    else                               data.activation_fn_id = 0;

    int n = static_cast<int>(nets.size());
    data.descs.resize(n);

    int total_nodes = 0, total_eval = 0, total_conn = 0, total_out = 0;

    for (int i = 0; i < n; ++i) {
        const auto& net        = nets[i];
        const auto& nodes      = net->raw_nodes();
        const auto& eval_order = net->eval_order();
        const auto& incoming   = net->incoming();
        const auto& nidx       = net->node_index_map();

        int num_conn = 0;
        for (auto nid : eval_order) {
            num_conn += static_cast<int>(incoming[nidx.at(nid)].size());
        }
        int num_out = 0;
        for (const auto& node : nodes) {
            if (node.type == NodeType::Output) ++num_out;
        }

        data.descs[i].num_nodes   = static_cast<int>(nodes.size());
        data.descs[i].num_eval    = static_cast<int>(eval_order.size());
        data.descs[i].num_inputs  = net->num_inputs();
        data.descs[i].num_outputs = net->num_outputs();
        data.descs[i].node_off    = total_nodes;
        data.descs[i].eval_off    = total_eval;
        data.descs[i].conn_off    = total_conn;
        data.descs[i].out_off     = total_out;

        total_nodes += static_cast<int>(nodes.size());
        total_eval  += static_cast<int>(eval_order.size());
        total_conn  += num_conn;
        total_out   += num_out;
    }

    data.node_types.resize(total_nodes);
    data.eval_order.resize(total_eval);
    data.conn_ptr.resize(total_eval);
    data.in_count.resize(total_eval);
    data.conn_from.resize(total_conn);
    data.conn_w.resize(total_conn);
    data.out_indices.resize(total_out);

    for (int i = 0; i < n; ++i) {
        const auto& net        = nets[i];
        const auto& nodes      = net->raw_nodes();
        const auto& eval_order = net->eval_order();
        const auto& incoming   = net->incoming();
        const auto& nidx       = net->node_index_map();
        const auto& desc       = data.descs[i];

        for (int j = 0; j < static_cast<int>(nodes.size()); ++j) {
            uint8_t t;
            switch (nodes[j].type) {
                case NodeType::Input:  t = 0; break;
                case NodeType::Bias:   t = 1; break;
                case NodeType::Hidden: t = 2; break;
                case NodeType::Output: t = 3; break;
                default:               t = 2; break;
            }
            data.node_types[desc.node_off + j] = t;
        }

        int conn_running = 0;
        for (int j = 0; j < static_cast<int>(eval_order.size()); ++j) {
            std::uint32_t nid = eval_order[j];
            int ni = nidx.at(nid);

            data.eval_order[desc.eval_off + j] = ni;
            data.conn_ptr  [desc.eval_off + j] = conn_running;
            data.in_count  [desc.eval_off + j] = static_cast<int>(incoming[ni].size());

            for (const auto& [from_idx, w] : incoming[ni]) {
                data.conn_from[desc.conn_off + conn_running] = from_idx;
                data.conn_w   [desc.conn_off + conn_running] = w;
                ++conn_running;
            }
        }

        int out_running = 0;
        for (int j = 0; j < static_cast<int>(nodes.size()); ++j) {
            if (nodes[j].type == NodeType::Output) {
                data.out_indices[desc.out_off + out_running] = j;
                ++out_running;
            }
        }
    }

    return data;
}

// ── Helper: build a fully-connected genome with known weights ───────────────
static Genome make_genome(int inputs, int outputs, float weight) {
    Genome g(inputs, outputs);
    std::uint32_t innov = 0;
    for (const auto& in_node : g.nodes()) {
        if (in_node.type != NodeType::Input && in_node.type != NodeType::Bias)
            continue;
        for (const auto& out_node : g.nodes()) {
            if (out_node.type != NodeType::Output) continue;
            g.add_connection({in_node.id, out_node.id, weight, true, innov++});
        }
    }
    return g;
}

static Genome make_hidden_genome() {
    Genome g(3, 2);
    InnovationTracker tracker;
    Random rng(42);

    for (const auto& in_node : g.nodes()) {
        if (in_node.type != NodeType::Input && in_node.type != NodeType::Bias) {
            continue;
        }
        for (const auto& out_node : g.nodes()) {
            if (out_node.type != NodeType::Output) {
                continue;
            }
            g.add_connection({in_node.id, out_node.id, rng.next_float(-1.0f, 1.0f), true,
                              tracker.get_innovation(in_node.id, out_node.id)});
        }
    }

    SimulationConfig config;
    config.mutation_rate = 1.0f;
    config.add_node_rate = 0.6f;
    config.add_connection_rate = 0.6f;
    for (int i = 0; i < 5; ++i) {
        Mutation::mutate(g, rng, config, tracker);
    }
    return g;
}

static Genome make_recurrent_genome() {
    Genome g(2, 1);
    g.add_node({4, NodeType::Hidden});
    g.add_connection({0, 4, 0.8f, true, 0});
    g.add_connection({1, 4, -0.4f, true, 1});
    g.add_connection({2, 4, 0.5f, true, 2});
    g.add_connection({4, 3, 1.1f, true, 3});
    g.add_connection({4, 4, 0.25f, true, 4});
    return g;
}

class GpuTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        gpu_available_ = init_cuda();
        if (gpu_available_) {
            print_device_info();
        }
    }
    void SetUp() override {
        if (!gpu_available_) GTEST_SKIP() << "No CUDA device available";
    }
    static bool gpu_available_;
};
bool GpuTest::gpu_available_ = false;

static void run_gpu_inference(GpuBatch& batch, const float* inputs, int input_count,
                              float* outputs, int output_count) {
    batch.pack_inputs_async(inputs, input_count);
    batch.launch_inference_async();
    batch.start_unpack_async();
    batch.finish_unpack(outputs, output_count);
    ASSERT_TRUE(batch.ok()) << "GPU batch reported a CUDA failure";
}

// Test 1: CPU vs GPU output comparison with known topology
TEST_F(GpuTest, CpuGpuOutputsMatch) {
    constexpr int kInputs = 3;
    constexpr int kOutputs = 2;
    constexpr float kWeight = 0.5f;

    Genome g = make_genome(kInputs, kOutputs, kWeight);
    std::vector<std::unique_ptr<NeuralNetwork>> nets;
    nets.push_back(std::make_unique<NeuralNetwork>(g, "sigmoid"));

    // CPU forward pass
    float cpu_in[kInputs] = {1.0f, 0.5f, -1.0f};
    float cpu_out[kOutputs] = {};
    nets[0]->activate_into(cpu_in, kInputs, cpu_out, kOutputs);

    // GPU forward pass
    GpuBatch batch(1, kInputs, kOutputs);
    auto data = build_test_network_data(nets, "sigmoid");
    batch.upload_network_data(data);
    ASSERT_TRUE(batch.ok()) << "GPU upload failed";

    float gpu_out[kOutputs] = {};
    run_gpu_inference(batch, cpu_in, kInputs, gpu_out, kOutputs);

    for (int i = 0; i < kOutputs; ++i) {
        EXPECT_NEAR(cpu_out[i], gpu_out[i], 1e-5f)
            << "Output " << i << " mismatch: CPU=" << cpu_out[i]
            << " GPU=" << gpu_out[i];
    }
}

// Test 2: Batch consistency — identical networks with identical inputs
TEST_F(GpuTest, BatchConsistency) {
    constexpr int kBatch = 100;
    constexpr int kInputs = 3;
    constexpr int kOutputs = 2;

    Genome g = make_genome(kInputs, kOutputs, 0.7f);
    std::vector<std::unique_ptr<NeuralNetwork>> nets;
    for (int i = 0; i < kBatch; ++i) {
        nets.push_back(std::make_unique<NeuralNetwork>(g, "sigmoid"));
    }

    GpuBatch batch(kBatch, kInputs, kOutputs);
    auto data = build_test_network_data(nets, "sigmoid");
    batch.upload_network_data(data);
    ASSERT_TRUE(batch.ok()) << "GPU upload failed";

    // All agents get same inputs
    std::vector<float> flat_in(kBatch * kInputs);
    for (int i = 0; i < kBatch; ++i) {
        flat_in[i * kInputs + 0] = 1.0f;
        flat_in[i * kInputs + 1] = 0.5f;
        flat_in[i * kInputs + 2] = -0.3f;
    }

    std::vector<float> flat_out(kBatch * kOutputs);
    run_gpu_inference(batch, flat_in.data(), kBatch * kInputs, flat_out.data(), kBatch * kOutputs);

    float cpu_out[kOutputs] = {};
    float cpu_in[kInputs] = {1.0f, 0.5f, -0.3f};
    nets[0]->activate_into(cpu_in, kInputs, cpu_out, kOutputs);

    for (int j = 0; j < kOutputs; ++j) {
        EXPECT_NEAR(cpu_out[j], flat_out[j], 1e-5f)
            << "Agent 0 output " << j << " mismatch: CPU=" << cpu_out[j]
            << " GPU=" << flat_out[j];
    }

    // All outputs should be identical
    for (int i = 1; i < kBatch; ++i) {
        for (int j = 0; j < kOutputs; ++j) {
            EXPECT_FLOAT_EQ(flat_out[j], flat_out[i * kOutputs + j])
                << "Agent " << i << " output " << j << " differs from agent 0";
        }
    }
}

// Test 3: Varied num_outputs — verify correct stride
TEST_F(GpuTest, VariedOutputCount) {
    for (int n_out : {1, 2, 3, 4}) {
        constexpr int kInputs = 2;
        constexpr int kBatch = 10;

        Genome g = make_genome(kInputs, n_out, 0.3f);
        std::vector<std::unique_ptr<NeuralNetwork>> nets;
        for (int i = 0; i < kBatch; ++i) {
            nets.push_back(std::make_unique<NeuralNetwork>(g, "sigmoid"));
        }

        GpuBatch batch(kBatch, kInputs, n_out);
        auto data = build_test_network_data(nets, "sigmoid");
        batch.upload_network_data(data);
        ASSERT_TRUE(batch.ok()) << "GPU upload failed";

        std::vector<float> flat_in(kBatch * kInputs, 0.5f);
        std::vector<float> flat_out(kBatch * n_out);

        run_gpu_inference(batch, flat_in.data(), kBatch * kInputs, flat_out.data(), kBatch * n_out);

        // CPU reference for agent 0
        float cpu_out[4] = {};
        float cpu_in[2] = {0.5f, 0.5f};
        nets[0]->activate_into(cpu_in, kInputs, cpu_out, n_out);

        for (int j = 0; j < n_out; ++j) {
            EXPECT_NEAR(cpu_out[j], flat_out[j], 1e-5f)
                << "n_out=" << n_out << " output " << j << " mismatch";
        }
    }
}

// Test 4: Zero-connection network (all connections disabled)
TEST_F(GpuTest, ZeroConnectionNetwork) {
    constexpr int kInputs = 3;
    constexpr int kOutputs = 2;

    // Create genome with no connections
    Genome g(kInputs, kOutputs);

    std::vector<std::unique_ptr<NeuralNetwork>> nets;
    nets.push_back(std::make_unique<NeuralNetwork>(g, "sigmoid"));

    GpuBatch batch(1, kInputs, kOutputs);
    auto data = build_test_network_data(nets, "sigmoid");
    batch.upload_network_data(data);
    ASSERT_TRUE(batch.ok()) << "GPU upload failed";

    float in_data[kInputs] = {1.0f, 0.5f, -1.0f};
    float out_data[kOutputs] = {};

    run_gpu_inference(batch, in_data, kInputs, out_data, kOutputs);

    // With no connections, outputs should be 0 (initialized) or sigmoid(0)
    // depending on whether output nodes are in eval_order with 0 incoming
    for (int i = 0; i < kOutputs; ++i) {
        EXPECT_TRUE(std::isfinite(out_data[i]))
            << "Output " << i << " is not finite: " << out_data[i];
    }
}

// Test 5: Large batch — verify no CUDA errors and finite outputs
TEST_F(GpuTest, LargeBatch) {
    constexpr int kBatch = 5000;
    constexpr int kInputs = 15;
    constexpr int kOutputs = 2;

    Genome g = make_genome(kInputs, kOutputs, 0.1f);
    std::vector<std::unique_ptr<NeuralNetwork>> nets;
    for (int i = 0; i < kBatch; ++i) {
        nets.push_back(std::make_unique<NeuralNetwork>(g, "sigmoid"));
    }

    GpuBatch batch(kBatch, kInputs, kOutputs);
    auto data = build_test_network_data(nets, "sigmoid");
    batch.upload_network_data(data);
    ASSERT_TRUE(batch.ok()) << "GPU upload failed";

    std::vector<float> flat_in(kBatch * kInputs);
    for (int i = 0; i < kBatch * kInputs; ++i) {
        flat_in[i] = static_cast<float>(i % 7) / 7.0f;  // deterministic pattern
    }

    std::vector<float> flat_out(kBatch * kOutputs);
    run_gpu_inference(batch, flat_in.data(), kBatch * kInputs, flat_out.data(), kBatch * kOutputs);

    for (int agent = 0; agent < kBatch; agent += 997) {
        float cpu_out[kOutputs] = {};
        nets[agent]->activate_into(flat_in.data() + agent * kInputs, kInputs, cpu_out, kOutputs);
        for (int j = 0; j < kOutputs; ++j) {
            EXPECT_NEAR(cpu_out[j], flat_out[agent * kOutputs + j], 1e-5f)
                << "Agent " << agent << " output " << j
                << " mismatch: CPU=" << cpu_out[j]
                << " GPU=" << flat_out[agent * kOutputs + j];
        }
    }

    for (int i = 0; i < kBatch * kOutputs; ++i) {
        EXPECT_TRUE(std::isfinite(flat_out[i]))
            << "Output at index " << i << " is not finite: " << flat_out[i];
    }
}

// Test 6: Activation function variants — tanh and relu produce correct results
TEST_F(GpuTest, ActivationFunctions) {
    for (const auto& fn : {"sigmoid", "tanh", "relu"}) {
        constexpr int kInputs = 3;
        constexpr int kOutputs = 2;

        Genome g = make_genome(kInputs, kOutputs, 0.5f);
        std::vector<std::unique_ptr<NeuralNetwork>> nets;
        nets.push_back(std::make_unique<NeuralNetwork>(g, fn));

        GpuBatch batch(1, kInputs, kOutputs);
        auto data = build_test_network_data(nets, fn);
        batch.upload_network_data(data);
        ASSERT_TRUE(batch.ok()) << "GPU upload failed";

        float in_data[kInputs] = {1.0f, -0.5f, 0.3f};
        float cpu_out[kOutputs] = {};
        nets[0]->activate_into(in_data, kInputs, cpu_out, kOutputs);

        float gpu_out[kOutputs] = {};
        run_gpu_inference(batch, in_data, kInputs, gpu_out, kOutputs);

        for (int i = 0; i < kOutputs; ++i) {
            EXPECT_NEAR(cpu_out[i], gpu_out[i], 1e-5f)
                << "Activation=" << fn << " output " << i
                << " mismatch: CPU=" << cpu_out[i] << " GPU=" << gpu_out[i];
        }
    }
}

TEST_F(GpuTest, HiddenTopologyMatchesCpu) {
    Genome g = make_hidden_genome();
    std::vector<std::unique_ptr<NeuralNetwork>> nets;
    nets.push_back(std::make_unique<NeuralNetwork>(g, "sigmoid"));

    GpuBatch batch(1, 3, 2);
    auto data = build_test_network_data(nets, "sigmoid");
    batch.upload_network_data(data);
    ASSERT_TRUE(batch.ok()) << "GPU upload failed";

    float in_data[3] = {0.8f, -0.25f, 0.5f};
    float cpu_out[2] = {};
    float gpu_out[2] = {};
    nets[0]->activate_into(in_data, 3, cpu_out, 2);
    run_gpu_inference(batch, in_data, 3, gpu_out, 2);

    for (int i = 0; i < 2; ++i) {
        EXPECT_NEAR(cpu_out[i], gpu_out[i], 1e-5f);
    }
}

TEST_F(GpuTest, RecurrentTopologyMatchesCpu) {
    Genome g = make_recurrent_genome();
    std::vector<std::unique_ptr<NeuralNetwork>> nets;
    nets.push_back(std::make_unique<NeuralNetwork>(g, "sigmoid"));

    GpuBatch batch(1, 2, 1);
    auto data = build_test_network_data(nets, "sigmoid");
    batch.upload_network_data(data);
    ASSERT_TRUE(batch.ok()) << "GPU upload failed";

    float in_data[2] = {0.2f, -0.7f};
    float cpu_out[1] = {};
    float gpu_out[1] = {};
    nets[0]->activate_into(in_data, 2, cpu_out, 1);
    run_gpu_inference(batch, in_data, 2, gpu_out, 1);

    EXPECT_NEAR(cpu_out[0], gpu_out[0], 1e-5f);
}

TEST_F(GpuTest, RejectsDescriptorCountMismatch) {
    Genome g = make_genome(3, 2, 0.5f);
    std::vector<std::unique_ptr<NeuralNetwork>> nets;
    nets.push_back(std::make_unique<NeuralNetwork>(g, "sigmoid"));

    GpuBatch batch(2, 3, 2);
    auto data = build_test_network_data(nets, "sigmoid");
    batch.upload_network_data(data);
    EXPECT_FALSE(batch.ok());
}

TEST_F(GpuTest, RejectsOversizedCopies) {
    Genome g = make_genome(3, 2, 0.5f);
    std::vector<std::unique_ptr<NeuralNetwork>> nets;
    nets.push_back(std::make_unique<NeuralNetwork>(g, "sigmoid"));

    GpuBatch batch(1, 3, 2);
    auto data = build_test_network_data(nets, "sigmoid");
    batch.upload_network_data(data);
    ASSERT_TRUE(batch.ok()) << "GPU upload failed";

    float oversized_inputs[4] = {1.0f, 0.5f, -1.0f, 0.0f};
    batch.pack_inputs_async(oversized_inputs, 4);
    EXPECT_FALSE(batch.ok());
}

TEST_F(GpuTest, EvaluateGenerationMatchesCpuAndCallbacks) {
    SimulationConfig config;
    config.predator_count = 400;
    config.prey_count = 700;
    config.generation_ticks = 12;
    config.seed = 42;
    config.target_fps = 30;

    Random cpu_rng(config.seed);
    Random gpu_rng(config.seed);
    SimulationManager cpu_sim(config);
    SimulationManager gpu_sim(config);
    EvolutionManager cpu_evo(config, cpu_rng);
    EvolutionManager gpu_evo(config, gpu_rng);

    cpu_sim.initialize();
    gpu_sim.initialize();
    cpu_evo.initialize(SensorInput::SIZE, 2);
    gpu_evo.initialize(SensorInput::SIZE, 2);

    int cpu_ticks = 0;
    int gpu_ticks = 0;
    cpu_evo.set_tick_callback([&](int, const SimulationManager&) { ++cpu_ticks; });
    gpu_evo.set_tick_callback([&](int, const SimulationManager&) { ++gpu_ticks; });
    gpu_evo.enable_gpu(true);
    ASSERT_TRUE(gpu_evo.gpu_enabled()) << "GPU path was not enabled for the parity test";

    cpu_evo.evaluate_generation(cpu_sim);
    gpu_evo.evaluate_generation(gpu_sim);

    EXPECT_EQ(cpu_ticks, gpu_ticks);
    EXPECT_EQ(cpu_sim.alive_predators(), gpu_sim.alive_predators());
    EXPECT_EQ(cpu_sim.alive_prey(), gpu_sim.alive_prey());

    for (size_t i = 0; i < cpu_evo.population().size(); i += 137) {
        EXPECT_NEAR(cpu_evo.population()[i].fitness(), gpu_evo.population()[i].fitness(), 1e-4f)
            << "Fitness mismatch at genome " << i;
    }
}

TEST_F(GpuTest, EvaluateGenerationMatchesCpuResidentHeadlessPath) {
    SimulationConfig config;
    config.predator_count = 250;
    config.prey_count = 750;
    config.food_count = 1000;
    config.generation_ticks = 4;
    config.seed = 1337;
    config.target_fps = 30;

    Random cpu_rng(config.seed);
    Random gpu_rng_a(config.seed);
    Random gpu_rng_b(config.seed);
    SimulationManager cpu_sim(config);
    SimulationManager gpu_sim_a(config);
    SimulationManager gpu_sim_b(config);
    EvolutionManager cpu_evo(config, cpu_rng);
    EvolutionManager gpu_evo_a(config, gpu_rng_a);
    EvolutionManager gpu_evo_b(config, gpu_rng_b);

    cpu_sim.initialize();
    gpu_sim_a.initialize();
    gpu_sim_b.initialize();
    cpu_evo.initialize(SensorInput::SIZE, 2);
    gpu_evo_a.initialize(SensorInput::SIZE, 2);
    gpu_evo_b.initialize(SensorInput::SIZE, 2);
    gpu_evo_a.enable_gpu(true);
    gpu_evo_b.enable_gpu(true);
    ASSERT_TRUE(gpu_evo_a.gpu_enabled()) << "GPU resident path was not enabled";
    ASSERT_TRUE(gpu_evo_b.gpu_enabled()) << "GPU resident path was not enabled";

    cpu_evo.evaluate_generation(cpu_sim);
    gpu_evo_a.evaluate_generation(gpu_sim_a);
    gpu_evo_b.evaluate_generation(gpu_sim_b);

    EXPECT_EQ(gpu_sim_a.alive_predators(), gpu_sim_b.alive_predators());
    EXPECT_EQ(gpu_sim_a.alive_prey(), gpu_sim_b.alive_prey());
    EXPECT_LE(std::abs(cpu_sim.alive_predators() - gpu_sim_a.alive_predators()), 1);
    EXPECT_LE(std::abs(cpu_sim.alive_prey() - gpu_sim_a.alive_prey()), 1);

    for (size_t i = 0; i < cpu_evo.population().size(); i += 17) {
        EXPECT_NEAR(gpu_evo_a.population()[i].fitness(), gpu_evo_b.population()[i].fitness(), 1e-4f)
            << "Resident deterministic fitness mismatch at genome " << i;
        EXPECT_NEAR(cpu_evo.population()[i].fitness(), gpu_evo_a.population()[i].fitness(), 1e-3f)
            << "Resident CPU/GPU fitness mismatch at genome " << i;
    }

    for (size_t i = 0; i < gpu_sim_a.agents().size(); i += 19) {
        const auto& gpu_agent_a = gpu_sim_a.agents()[i];
        const auto& gpu_agent_b = gpu_sim_b.agents()[i];
        EXPECT_EQ(gpu_agent_a->alive(), gpu_agent_b->alive()) << "Resident alive mismatch at agent " << i;
        EXPECT_NEAR(gpu_agent_a->position().x, gpu_agent_b->position().x, 1e-5f);
        EXPECT_NEAR(gpu_agent_a->position().y, gpu_agent_b->position().y, 1e-5f);
        EXPECT_NEAR(gpu_agent_a->energy(), gpu_agent_b->energy(), 1e-5f);
        EXPECT_EQ(gpu_agent_a->kills(), gpu_agent_b->kills());
        EXPECT_EQ(gpu_agent_a->food_eaten(), gpu_agent_b->food_eaten());
    }

    for (size_t i = 0; i < cpu_sim.agents().size(); i += 19) {
        const auto& cpu_agent = cpu_sim.agents()[i];
        const auto& gpu_agent = gpu_sim_a.agents()[i];
        EXPECT_NEAR(cpu_agent->position().x, gpu_agent->position().x, 1.0f);
        EXPECT_NEAR(cpu_agent->position().y, gpu_agent->position().y, 1.0f);
        EXPECT_NEAR(cpu_agent->energy(), gpu_agent->energy(), 1.0f);
    }
}
