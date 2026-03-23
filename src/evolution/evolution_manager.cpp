#include "evolution/evolution_manager.hpp"
#include "evolution/crossover.hpp"
#include "core/profiler.hpp"
#include "simulation/physics.hpp"
#include "core/lua_runtime.hpp"

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unordered_map>
#include <chrono>
#include <fstream>
#include <sstream>

#ifdef MOONAI_OPENMP_ENABLED
#include <omp.h>
#endif

#ifdef MOONAI_ENABLE_CUDA
#include "gpu/gpu_batch.hpp"

namespace {

constexpr int kGpuMinPopulation = 1000;

}

// ── GPU network packing helper ────────────────────────────────────────────────
// Extracts the CSR-packed flat arrays from NeuralNetwork objects into the
// GpuNetworkData struct. Lives here (in moonai_evolution) because it needs
// NeuralNetwork internals, keeping moonai_gpu free of moonai_evolution dependency.
static moonai::gpu::GpuNetworkData build_gpu_network_data(
    const std::vector<std::unique_ptr<moonai::NeuralNetwork>>& nets,
    const std::string& activation_fn)
{
    moonai::gpu::GpuNetworkData data;

    if (activation_fn == "tanh")       data.activation_fn_id = 1;
    else if (activation_fn == "relu")  data.activation_fn_id = 2;
    else                               data.activation_fn_id = 0;

    int n = static_cast<int>(nets.size());
    data.descs.resize(n);

    int total_nodes = 0, total_eval = 0, total_conn = 0, total_out = 0;

    // Pass 1: compute cumulative sizes and fill desc offsets
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
            if (node.type == moonai::NodeType::Output) ++num_out;
        }

        data.descs[i].num_nodes  = static_cast<int>(nodes.size());
        data.descs[i].num_eval   = static_cast<int>(eval_order.size());
        data.descs[i].num_inputs  = net->num_inputs();
        data.descs[i].num_outputs = net->num_outputs();
        data.descs[i].node_off   = total_nodes;
        data.descs[i].eval_off   = total_eval;
        data.descs[i].conn_off   = total_conn;
        data.descs[i].out_off    = total_out;

        total_nodes += static_cast<int>(nodes.size());
        total_eval  += static_cast<int>(eval_order.size());
        total_conn  += num_conn;
        total_out   += num_out;
    }

    // Allocate flat host arrays
    data.node_types.resize(total_nodes);
    data.eval_order.resize(total_eval);
    data.conn_ptr.resize(total_eval);
    data.in_count.resize(total_eval);
    data.conn_from.resize(total_conn);
    data.conn_w.resize(total_conn);
    data.out_indices.resize(total_out);

    // Pass 2: fill flat arrays
    for (int i = 0; i < n; ++i) {
        const auto& net        = nets[i];
        const auto& nodes      = net->raw_nodes();
        const auto& eval_order = net->eval_order();
        const auto& incoming   = net->incoming();
        const auto& nidx       = net->node_index_map();
        const auto& desc       = data.descs[i];

        // node_types: 0=Input, 1=Bias, 2=Hidden, 3=Output
        for (int j = 0; j < static_cast<int>(nodes.size()); ++j) {
            uint8_t t;
            switch (nodes[j].type) {
                case moonai::NodeType::Input:  t = 0; break;
                case moonai::NodeType::Bias:   t = 1; break;
                case moonai::NodeType::Hidden: t = 2; break;
                case moonai::NodeType::Output: t = 3; break;
                default:                       t = 2; break;
            }
            data.node_types[desc.node_off + j] = t;
        }

        // eval_order, conn_ptr, in_count, conn_from, conn_w
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

        // out_indices
        int out_running = 0;
        for (int j = 0; j < static_cast<int>(nodes.size()); ++j) {
            if (nodes[j].type == moonai::NodeType::Output) {
                data.out_indices[desc.out_off + out_running] = j;
                ++out_running;
            }
        }
    }

    return data;
}
#endif

namespace moonai {

EvolutionManager::EvolutionManager(const SimulationConfig& config, Random& rng)
    : config_(config)
    , rng_(rng) {
}

EvolutionManager::~EvolutionManager() = default;

void EvolutionManager::initialize(int num_inputs, int num_outputs) {
    population_.clear();
    species_.clear();
    generation_ = 0;
    num_inputs_ = num_inputs;
    num_outputs_ = num_outputs;

    int total_pop = config_.predator_count + config_.prey_count;

    // Create initial population with fully-connected minimal topologies
    for (int i = 0; i < total_pop; ++i) {
        Genome g(num_inputs, num_outputs);
        // Connect all inputs (+ bias) to all outputs with random weights
        for (const auto& in_node : g.nodes()) {
            if (in_node.type != NodeType::Input && in_node.type != NodeType::Bias)
                continue;
            for (const auto& out_node : g.nodes()) {
                if (out_node.type != NodeType::Output) continue;
                std::uint32_t innov = tracker_.get_innovation(in_node.id, out_node.id);
                g.add_connection({in_node.id, out_node.id,
                                  rng_.next_float(-1.0f, 1.0f),
                                  true, innov});
            }
        }
        population_.push_back(std::move(g));
    }

    build_networks();

    spdlog::info("Evolution initialized: {} genomes ({} inputs, {} outputs)",
                 total_pop, num_inputs, num_outputs);
}

void EvolutionManager::enable_gpu(bool use_gpu) {
    use_gpu_ = use_gpu;
#ifdef MOONAI_ENABLE_CUDA
    if (use_gpu_) {
        int total_agents = config_.predator_count + config_.prey_count;
        if (total_agents < kGpuMinPopulation) {
            spdlog::info("GPU disabled: population {} < {} threshold "
                         "(kernel launch overhead exceeds computation)",
                         total_agents, kGpuMinPopulation);
            use_gpu_ = false;
            return;
        }
        if (num_inputs_ > 0 && !population_.empty()) {
            gpu_batch_ = std::make_unique<moonai::gpu::GpuBatch>(
                static_cast<int>(population_.size()), num_inputs_, num_outputs_);
            spdlog::debug("GpuBatch allocated: {} agents, {} inputs, {} outputs",
                          population_.size(), num_inputs_, num_outputs_);
            if (!gpu_batch_->ok()) {
                spdlog::warn("GpuBatch allocation failed; falling back to CPU inference");
                gpu_batch_.reset();
                use_gpu_ = false;
            }
        }
    }
#endif
}

#ifdef MOONAI_ENABLE_CUDA
bool EvolutionManager::prepare_gpu_generation() {
    ScopedTimer timer(ProfileEvent::PrepareGpuGeneration);
    if (!use_gpu_ || !gpu_batch_) {
        return false;
    }

    gpu_batch_->upload_network_data(
        build_gpu_network_data(networks_, config_.activation_function));
    if (!gpu_batch_->ok()) {
        spdlog::warn("GPU network upload failed; falling back to CPU inference");
        gpu_batch_.reset();
        use_gpu_ = false;
        return false;
    }
    return true;
}

bool EvolutionManager::infer_actions_gpu(const SimulationManager& sim, std::vector<Vec2>& actions) {
    if (!use_gpu_ || !gpu_batch_) {
        return false;
    }

    int agent_count = static_cast<int>(std::min(sim.agents().size(), networks_.size()));
    if (agent_count <= 0) {
        return true;
    }

    if (static_cast<int>(actions.size()) < agent_count) {
        actions.resize(agent_count, {0.0f, 0.0f});
    }
    std::fill(actions.begin(), actions.end(), Vec2{0.0f, 0.0f});

    {
        ScopedTimer timer(ProfileEvent::GpuSensorFlatten);
        sim.write_sensors_flat(gpu_batch_->host_inputs(), static_cast<size_t>(agent_count));
    }

    const int in_count = agent_count * num_inputs_;
    {
        ScopedTimer timer(ProfileEvent::GpuPackInputs);
        gpu_batch_->pack_inputs_async(in_count);
    }
    {
        ScopedTimer timer(ProfileEvent::GpuLaunch);
        gpu_batch_->launch_inference_async();
    }
    {
        ScopedTimer timer(ProfileEvent::GpuStartUnpack);
        gpu_batch_->start_unpack_async();
    }
    {
        ScopedTimer timer(ProfileEvent::GpuFinishUnpack);
        gpu_batch_->finish_unpack();
    }

    if (!gpu_batch_->ok()) {
        spdlog::warn("GPU inference failed; falling back to CPU inference");
        gpu_batch_.reset();
        use_gpu_ = false;
        return false;
    }

    {
        ScopedTimer timer(ProfileEvent::GpuOutputConvert);
        const float* flat_out = gpu_batch_->host_outputs();
        for (int i = 0; i < agent_count; ++i) {
            if (!sim.agents()[i]->alive()) {
                continue;
            }
            if (num_outputs_ >= 2) {
                actions[i] = Vec2{
                    flat_out[i * num_outputs_ + 0] * 2.0f - 1.0f,
                    flat_out[i * num_outputs_ + 1] * 2.0f - 1.0f
                };
            }
        }
    }
    return true;
}
#endif

void EvolutionManager::evaluate_generation(SimulationManager& sim) {
    build_networks();

    float dt = 1.0f / static_cast<float>(config_.target_fps);
    int last_tick = 0;
    int agent_count = static_cast<int>(std::min(sim.agents().size(), networks_.size()));
    std::vector<Vec2> actions(sim.agents().size(), {0.0f, 0.0f});

    int max_threads = 1;
#ifdef MOONAI_OPENMP_ENABLED
    max_threads = omp_get_max_threads();
#endif
    std::vector<std::vector<float>> thread_sensor_bufs(max_threads,
        std::vector<float>(SensorInput::SIZE));
    std::vector<std::vector<float>> thread_output_bufs(max_threads,
        std::vector<float>(num_outputs_));

#ifdef MOONAI_ENABLE_CUDA
    if (prepare_gpu_generation()) {
        Profiler::instance().mark_gpu_used(true);
        for (int tick = 0; tick < config_.generation_ticks; ++tick) {
            if (!infer_actions_gpu(sim, actions)) {
                break;
            }

            {
                ScopedTimer timer(ProfileEvent::ApplyActions);
                for (int i = 0; i < agent_count; ++i) {
                    if (!sim.agents()[i]->alive()) continue;
                    sim.apply_action(static_cast<size_t>(i), actions[i], dt);
                }
            }

            sim.tick(dt);
            last_tick = tick + 1;
            Profiler::instance().increment(ProfileCounter::TicksExecuted);

            if (tick_callback_) {
                ScopedTimer timer(ProfileEvent::TickCallback);
                tick_callback_(tick, sim);
            }
            if (sim.alive_prey() == 0 || sim.alive_predators() == 0) break;
        }

        if (use_gpu_) {
            compute_fitness(sim);
            return;
        }
    }
#endif

    // CPU path (OpenMP-parallelized)
    ScopedTimer cpu_eval_timer(ProfileEvent::CpuEvalTotal);
    for (int tick = last_tick; tick < config_.generation_ticks; ++tick) {
        // Parallel compute phase: sensor + NN per agent (read-only on shared state)
        std::fill(actions.begin(), actions.end(), Vec2{0.0f, 0.0f});

        #pragma omp parallel for schedule(dynamic) if(MOONAI_OPENMP_ENABLED)
        for (int i = 0; i < agent_count; ++i) {
            if (!sim.agents()[i]->alive()) continue;

            int tid = 0;
#ifdef MOONAI_OPENMP_ENABLED
            tid = omp_get_thread_num();
#endif
            float* sensor_buf = thread_sensor_bufs[tid].data();
            float* output_buf = thread_output_bufs[tid].data();

            if (Profiler::instance().enabled()) {
                const auto sensor_start = std::chrono::steady_clock::now();
                sim.get_sensors(static_cast<size_t>(i)).write_to(sensor_buf);
                const auto sensor_end = std::chrono::steady_clock::now();
                Profiler::instance().add_duration(
                    ProfileEvent::CpuSensorBuild,
                    std::chrono::duration_cast<std::chrono::nanoseconds>(sensor_end - sensor_start).count());

                const auto nn_start = std::chrono::steady_clock::now();
                networks_[i]->activate_into(sensor_buf, num_inputs_,
                                            output_buf, num_outputs_);
                const auto nn_end = std::chrono::steady_clock::now();
                Profiler::instance().add_duration(
                    ProfileEvent::CpuNnActivate,
                    std::chrono::duration_cast<std::chrono::nanoseconds>(nn_end - nn_start).count());
            } else {
                sim.get_sensors(static_cast<size_t>(i)).write_to(sensor_buf);
                networks_[i]->activate_into(sensor_buf, num_inputs_,
                                            output_buf, num_outputs_);
            }

            if (num_outputs_ >= 2) {
                actions[i] = Vec2{
                    output_buf[0] * 2.0f - 1.0f,
                    output_buf[1] * 2.0f - 1.0f
                };
            }
        }

        // Sequential apply phase
        {
            ScopedTimer timer(ProfileEvent::ApplyActions);
            for (size_t i = 0; i < sim.agents().size(); ++i) {
                if (!sim.agents()[i]->alive()) continue;
                sim.apply_action(i, actions[i], dt);
            }
        }

        sim.tick(dt);
        last_tick = tick + 1;
        Profiler::instance().increment(ProfileCounter::TicksExecuted);

        if (tick_callback_) {
            ScopedTimer timer(ProfileEvent::TickCallback);
            tick_callback_(tick, sim);
        }

        // Early exit if all prey or all predators are dead
        if (sim.alive_prey() == 0 || sim.alive_predators() == 0) {
            break;
        }
    }
    // Compute fitness from simulation results
    compute_fitness(sim);
}

void EvolutionManager::evolve() {
    tracker_.reset_generation();

    speciate();

    // Dynamic compatibility threshold (Stanley 2002, Section 3.3):
    // Adjust delta_t to maintain a target number of species.
    int target_species = std::max(2, (config_.predator_count + config_.prey_count) / 10);
    if (static_cast<int>(species_.size()) > target_species)
        config_.compatibility_threshold += 0.3f;
    else if (static_cast<int>(species_.size()) < target_species)
        config_.compatibility_threshold -= 0.3f;
    config_.compatibility_threshold = std::max(config_.compatibility_threshold, 0.5f);

    remove_stagnant_species();
    reproduce();

    ++generation_;

    spdlog::debug("Generation {}: {} species, {} genomes, threshold={:.2f}",
                  generation_, species_.size(), population_.size(),
                  config_.compatibility_threshold);
}

void EvolutionManager::build_networks() {
    ScopedTimer timer(ProfileEvent::BuildNetworks);
    networks_.clear();
    networks_.reserve(population_.size());
    for (const auto& genome : population_) {
        networks_.push_back(std::make_unique<NeuralNetwork>(genome, config_.activation_function));
    }
}

void EvolutionManager::compute_fitness(const SimulationManager& sim) {
    ScopedTimer timer(ProfileEvent::ComputeFitness);
    const auto& agents = sim.agents();
    bool use_lua = lua_runtime_ && lua_runtime_->callbacks().has_fitness_fn;

    for (size_t i = 0; i < population_.size() && i < agents.size(); ++i) {
        const auto& agent = agents[i];

        // Pre-compute normalized stats (used by both Lua and default paths)
        float age_ratio = static_cast<float>(agent->age()) /
                          static_cast<float>(config_.generation_ticks);
        float kills_or_food = (agent->type() == AgentType::Predator)
            ? static_cast<float>(agent->kills())
            : static_cast<float>(agent->food_eaten());
        float energy_ratio = std::max(0.0f, agent->energy()) / config_.initial_energy;
        float alive_bonus  = agent->alive() ? 1.0f : 0.0f;
        float max_dist = agent->speed()
            * static_cast<float>(config_.generation_ticks)
            / static_cast<float>(config_.target_fps);
        float dist_ratio = (max_dist > 0.0f)
            ? std::min(agent->distance_traveled() / max_dist, 1.0f)
            : 0.0f;
        float complexity = static_cast<float>(population_[i].complexity());

        float fitness;
        if (use_lua) {
            fitness = lua_runtime_->call_fitness(
                age_ratio, kills_or_food, energy_ratio,
                alive_bonus, dist_ratio, complexity, config_);
        } else {
            fitness = default_fitness(age_ratio, kills_or_food, energy_ratio,
                                      alive_bonus, dist_ratio, complexity);
        }

        population_[i].set_fitness(fitness);
    }
}

float EvolutionManager::default_fitness(float age_ratio, float kills_or_food,
    float energy_ratio, float alive_bonus, float dist_ratio, float complexity) const {
    float f = config_.fitness_survival_weight * age_ratio
            + config_.fitness_kill_weight     * kills_or_food
            + config_.fitness_energy_weight   * energy_ratio
            + alive_bonus
            + config_.fitness_distance_weight * dist_ratio
            - config_.complexity_penalty_weight * complexity;
    return std::max(0.0f, f);
}

void EvolutionManager::speciate() {
    ScopedTimer timer(ProfileEvent::Speciate);
    // Clear stale member pointers first (they point into the previous generation's
    // population which was freed in reproduce()), then update the representative
    // from the now-empty members (keeps the existing representative if no members).
    for (auto& s : species_) {
        s.clear_members();
        s.update_representative();
    }

    // Assign each genome to a species
    for (auto& genome : population_) {
        bool placed = false;
        for (auto& s : species_) {
            if (s.is_compatible(genome, config_.compatibility_threshold,
                                config_.c1_excess, config_.c2_disjoint, config_.c3_weight)) {
                s.add_member(&genome);
                placed = true;
                break;
            }
        }
        if (!placed) {
            species_.emplace_back(genome);
            species_.back().add_member(&genome);
        }
    }

    // Remove empty species
    species_.erase(
        std::remove_if(species_.begin(), species_.end(),
                        [](const Species& s) { return s.members().empty(); }),
        species_.end());

    // Adjust fitness and track stagnation
    for (auto& s : species_) {
        s.adjust_fitness();
        s.sort_by_fitness();
        s.update_best_fitness();
    }
}

void EvolutionManager::remove_stagnant_species() {
    ScopedTimer timer(ProfileEvent::RemoveStagnantSpecies);
    if (species_.size() <= 2) return;  // Always keep at least 2 species

    species_.erase(
        std::remove_if(species_.begin(), species_.end(),
            [this](const Species& s) {
                return s.is_stagnant(config_.stagnation_limit);
            }),
        species_.end());

    // Ensure at least one species remains
    if (species_.empty() && !population_.empty()) {
        species_.emplace_back(population_[0]);
        for (auto& g : population_) {
            species_.back().add_member(&g);
        }
        species_.back().adjust_fitness();
        species_.back().sort_by_fitness();
    }
}

void EvolutionManager::reproduce() {
    ScopedTimer timer(ProfileEvent::Reproduce);
    std::vector<Genome> new_population;
    int target_size = static_cast<int>(population_.size());

    if (species_.empty()) {
        // Fallback: re-initialize population
        spdlog::warn("All species extinct. Re-initializing population.");
        initialize(num_inputs_, num_outputs_);
        return;
    }

    // Calculate total adjusted fitness across all species
    float total_adjusted = 0.0f;
    for (const auto& s : species_) {
        total_adjusted += s.total_adjusted_fitness();
    }

    // Largest-remainder method: guarantees total == target_size exactly
    std::vector<int> quotas(species_.size(), 1);  // min 1 per species
    int allocated = static_cast<int>(species_.size());

    if (total_adjusted > 0.0f && allocated < target_size) {
        std::vector<float> remainders(species_.size(), 0.0f);
        for (size_t i = 0; i < species_.size(); ++i) {
            float exact = species_[i].total_adjusted_fitness() / total_adjusted
                          * static_cast<float>(target_size);
            int floor_val = std::max(1, static_cast<int>(exact));
            quotas[i] = floor_val;
            remainders[i] = exact - static_cast<float>(floor_val);
            allocated += floor_val - 1;  // already counted the min-1
        }
        // Distribute leftover slots by largest remainder
        int leftover = target_size - allocated;
        std::vector<size_t> order(species_.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
            [&](size_t a, size_t b) { return remainders[a] > remainders[b]; });
        for (int k = 0; k < leftover && k < static_cast<int>(order.size()); ++k) {
            quotas[order[k]]++;
        }
    }

    // Produce offspring for each species according to its quota
    for (size_t si = 0; si < species_.size(); ++si) {
        auto& s = species_[si];
        const auto& members = s.members();
        if (members.empty()) continue;

        int offspring_count = quotas[si];

        // Elitism: keep the best member unchanged
        new_population.push_back(*members[0]);

        // Produce remaining offspring via crossover + mutation
        for (int i = 1; i < offspring_count; ++i) {
            Genome child;

            if (members.size() == 1 || !rng_.next_bool(config_.crossover_rate)) {
                // Mutation only
                child = *members[rng_.next_int(0, static_cast<int>(members.size()) - 1)];
            } else {
                // Crossover + mutation
                int p1 = rng_.next_int(0, static_cast<int>(members.size()) - 1);
                int p2 = rng_.next_int(0, static_cast<int>(members.size()) - 1);
                child = Crossover::crossover(*members[p1], *members[p2], rng_);
            }

            Mutation::mutate(child, rng_, config_, tracker_);
            new_population.push_back(std::move(child));
        }
    }

    // Trim to target (elitism can slightly overshoot if quota rounds up)
    if (static_cast<int>(new_population.size()) > target_size) {
        new_population.resize(target_size);
    }

    population_ = std::move(new_population);
}

void EvolutionManager::save_checkpoint(const std::string& path, const Random& rng) const {
    nlohmann::json j;
    j["format_version"] = 1;
    j["generation"] = generation_;
    j["compatibility_threshold"] = config_.compatibility_threshold;
    j["innovation_counter"] = tracker_.innovation_count();
    j["node_counter"] = tracker_.node_count();
    j["species_next_id"] = Species::next_species_id();  // peek current counter

    // Serialize RNG state
    std::ostringstream oss;
    oss << const_cast<Random&>(rng).engine();
    j["rng_state"] = oss.str();

    // Serialize population (genome + species_id)
    j["population"] = nlohmann::json::array();
    for (const auto& genome : population_) {
        nlohmann::json entry;
        entry["species_id"] = -1;
        // Find which species this genome belongs to
        for (const auto& s : species_) {
            for (const auto* m : s.members()) {
                if (m == &genome) {
                    entry["species_id"] = s.id();
                    break;
                }
            }
            if (entry["species_id"] != -1) break;
        }
        entry["genome"] = nlohmann::json::parse(genome.to_json());
        j["population"].push_back(entry);
    }

    // Serialize species metadata
    j["species"] = nlohmann::json::array();
    for (const auto& s : species_) {
        nlohmann::json se;
        se["id"] = s.id();
        se["representative"] = nlohmann::json::parse(s.representative().to_json());
        se["best_fitness_ever"] = s.best_fitness_ever();
        se["generations_without_improvement"] = s.generations_without_improvement();
        j["species"].push_back(se);
    }

    std::ofstream file(path);
    if (!file.is_open()) {
        spdlog::error("Cannot write checkpoint to '{}'", path);
        return;
    }
    file << j.dump(2);
    spdlog::info("Checkpoint saved: {} (gen {})", path, generation_);
}

bool EvolutionManager::load_checkpoint(const std::string& path, Random& rng) {
    std::ifstream file(path);
    if (!file.is_open()) {
        spdlog::error("Cannot open checkpoint '{}'", path);
        return false;
    }

    nlohmann::json j;
    try {
        file >> j;
    } catch (const nlohmann::json::parse_error& e) {
        spdlog::error("Failed to parse checkpoint '{}': {}", path, e.what());
        return false;
    }

    generation_ = j["generation"];
    config_.compatibility_threshold = j["compatibility_threshold"];

    std::uint32_t innov = j["innovation_counter"];
    std::uint32_t node  = j["node_counter"];
    tracker_.set_counters(innov, node);

    int species_next = j["species_next_id"];
    Species::set_next_species_id(species_next);

    // Restore RNG state
    std::string rng_str = j["rng_state"];
    std::istringstream iss(rng_str);
    iss >> rng.engine();

    // Restore species (without members; members rebuilt below)
    species_.clear();
    for (const auto& se : j["species"]) {
        Genome rep = Genome::from_json(se["representative"].dump());
        Species s(rep);
        // Override id (Species constructor auto-increments, but we set next_id above)
        // We need to reconstruct with the correct id — create and then fix by building
        // a matching species entry. Since set_next_species_id was called before loop,
        // we already incremented; we need to handle ids by rebuilding species_ from scratch.
        // The simplest approach: build species map for later lookup.
        (void)se["best_fitness_ever"].get<float>();   // validate fields exist
        species_.push_back(std::move(s));
    }
    // Re-do species: species_ was built above with auto-incremented IDs.
    // Rebuild properly using saved IDs by restarting from a fresh counter.
    // The set_next_species_id call above set counter to species_next, so the
    // Species constructed in the loop used IDs >= species_next. That's wrong.
    // Fix: set counter to the first saved id before building, clear and rebuild.
    species_.clear();
    for (const auto& se : j["species"]) {
        int saved_id = se["id"];
        Species::set_next_species_id(saved_id);
        Genome rep = Genome::from_json(se["representative"].dump());
        Species s(rep);  // will get id == saved_id
        species_.push_back(std::move(s));
    }
    // Restore stagnation metadata on each species
    for (size_t si = 0; si < species_.size() && si < j["species"].size(); ++si) {
        const auto& se = j["species"][si];
        species_[si].restore_stagnation(
            se["best_fitness_ever"].get<float>(),
            se["generations_without_improvement"].get<int>());
    }
    // Restore the counter to species_next_id so new species get correct IDs
    Species::set_next_species_id(species_next);

    // Restore population
    population_.clear();
    std::vector<int> genome_species_ids;
    for (const auto& entry : j["population"]) {
        population_.push_back(Genome::from_json(entry["genome"].dump()));
        genome_species_ids.push_back(entry["species_id"].get<int>());
    }

    // Rebuild species members by matching saved species_id
    for (size_t gi = 0; gi < population_.size(); ++gi) {
        int sid = genome_species_ids[gi];
        for (auto& s : species_) {
            if (s.id() == sid) {
                s.add_member(&population_[gi]);
                break;
            }
        }
    }

    build_networks();

    spdlog::info("Checkpoint loaded: '{}' (resuming from gen {})", path, generation_);
    return true;
}

const Genome* EvolutionManager::genome_at(int idx) const {
    if (idx < 0 || idx >= static_cast<int>(population_.size())) return nullptr;
    return &population_[idx];
}

void EvolutionManager::assign_species_ids(SimulationManager& sim) const {
    const auto& agents = sim.agents();
    // Build genome pointer → species ID map
    std::unordered_map<const Genome*, int> genome_species;
    for (const auto& s : species_) {
        for (const auto* g : s.members()) {
            genome_species[g] = s.id();
        }
    }
    // Population index matches agent index
    for (size_t i = 0; i < population_.size() && i < agents.size(); ++i) {
        auto it = genome_species.find(&population_[i]);
        if (it != genome_species.end()) {
            agents[i]->set_species_id(it->second);
        }
    }
}

} // namespace moonai
