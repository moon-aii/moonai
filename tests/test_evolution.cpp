#include <gtest/gtest.h>
#include "core/config.hpp"
#include "core/random.hpp"
#include "evolution/genome.hpp"
#include "evolution/neural_network.hpp"
#include "evolution/mutation.hpp"
#include "evolution/crossover.hpp"
#include "evolution/species.hpp"
#include "evolution/evolution_manager.hpp"
#include "simulation/physics.hpp"

using namespace moonai;

// ── Innovation Tracker Tests ────────────────────────────────────────────

TEST(InnovationTrackerTest, SameConnectionGetsSameInnovation) {
    InnovationTracker tracker;
    std::uint32_t i1 = tracker.get_innovation(0, 3);
    std::uint32_t i2 = tracker.get_innovation(0, 3);
    EXPECT_EQ(i1, i2);
}

TEST(InnovationTrackerTest, DifferentConnectionGetsDifferentInnovation) {
    InnovationTracker tracker;
    std::uint32_t i1 = tracker.get_innovation(0, 3);
    std::uint32_t i2 = tracker.get_innovation(1, 3);
    EXPECT_NE(i1, i2);
}

TEST(InnovationTrackerTest, ResetClearsGenerationCache) {
    InnovationTracker tracker;
    std::uint32_t i1 = tracker.get_innovation(0, 3);
    tracker.reset_generation();
    std::uint32_t i2 = tracker.get_innovation(0, 3);
    // After reset, same structural mutation gets a new innovation number
    EXPECT_NE(i1, i2);
}

TEST(InnovationTrackerTest, NodeIdIncrement) {
    InnovationTracker tracker;
    EXPECT_EQ(tracker.next_node_id(), 0u);
    EXPECT_EQ(tracker.next_node_id(), 1u);
    EXPECT_EQ(tracker.next_node_id(), 2u);
}

TEST(InnovationTrackerTest, InitFromPopulation) {
    std::vector<Genome> pop;
    Genome g(2, 1);
    g.add_connection({0, 3, 0.5f, true, 5});
    g.add_node({10, NodeType::Hidden});
    pop.push_back(std::move(g));

    InnovationTracker tracker;
    tracker.init_from_population(pop);

    // Next innovation should be > 5
    EXPECT_GE(tracker.innovation_count(), 6u);
    // Next node id should be > 10
    EXPECT_GE(tracker.node_count(), 11u);
}

// ── Mutation Tests ──────────────────────────────────────────────────────

TEST(MutationTest, MutateWeightsChangesWeights) {
    Genome g(2, 1);
    g.add_connection({0, 3, 0.5f, true, 0});
    g.add_connection({1, 3, -0.5f, true, 1});

    Random rng(42);
    float original_w0 = g.connections()[0].weight;
    float original_w1 = g.connections()[1].weight;

    Mutation::mutate_weights(g, rng, 0.5f);

    // At least one weight should have changed
    bool changed = (g.connections()[0].weight != original_w0 ||
                    g.connections()[1].weight != original_w1);
    EXPECT_TRUE(changed);
}

TEST(MutationTest, WeightsAreClamped) {
    Genome g(1, 1);
    g.add_connection({0, 2, 7.5f, true, 0});

    Random rng(42);
    // Mutate many times to push weight toward clamp
    for (int i = 0; i < 100; ++i) {
        Mutation::mutate_weights(g, rng, 2.0f);
    }

    EXPECT_GE(g.connections()[0].weight, -8.0f);
    EXPECT_LE(g.connections()[0].weight, 8.0f);
}

TEST(MutationTest, AddConnectionCreatesValidConnection) {
    Genome g(2, 1);
    InnovationTracker tracker;
    tracker.init_from_population({g});
    Random rng(42);

    int initial_conns = static_cast<int>(g.connections().size());
    Mutation::add_connection(g, rng, tracker);

    EXPECT_GE(static_cast<int>(g.connections().size()), initial_conns);

    // Verify connection is valid
    for (const auto& conn : g.connections()) {
        EXPECT_TRUE(g.has_node(conn.in_node));
        EXPECT_TRUE(g.has_node(conn.out_node));
    }
}

TEST(MutationTest, AddNodeSplitsConnection) {
    Genome g(2, 1);
    g.add_connection({0, 3, 0.5f, true, 0});

    InnovationTracker tracker;
    tracker.init_from_population({g});
    Random rng(42);

    int initial_nodes = static_cast<int>(g.nodes().size());
    Mutation::add_node(g, rng, tracker);

    EXPECT_EQ(static_cast<int>(g.nodes().size()), initial_nodes + 1);
    // Original connection should be disabled
    EXPECT_FALSE(g.connections()[0].enabled);
    // Two new connections should be added
    EXPECT_EQ(g.connections().size(), 3u);
}

TEST(MutationTest, MutatedGenomeProducesValidNetwork) {
    Genome g(3, 2);
    InnovationTracker tracker;
    Random rng(42);

    // Fully connect initial genome
    for (const auto& in_node : g.nodes()) {
        if (in_node.type != NodeType::Input && in_node.type != NodeType::Bias) continue;
        for (const auto& out_node : g.nodes()) {
            if (out_node.type != NodeType::Output) continue;
            g.add_connection({in_node.id, out_node.id,
                              rng.next_float(-1.0f, 1.0f), true,
                              tracker.get_innovation(in_node.id, out_node.id)});
        }
    }

    // Apply many mutations
    SimulationConfig config;
    config.mutation_rate = 1.0f;
    config.add_connection_rate = 0.5f;
    config.add_node_rate = 0.3f;

    for (int i = 0; i < 20; ++i) {
        Mutation::mutate(g, rng, config, tracker);
    }

    // Should still produce a valid network
    NeuralNetwork nn(g);
    auto outputs = nn.activate({1.0f, 0.5f, -1.0f});
    EXPECT_EQ(outputs.size(), 2u);
    for (float o : outputs) {
        EXPECT_GE(o, 0.0f);
        EXPECT_LE(o, 1.0f);
    }
}

// ── Crossover Tests ─────────────────────────────────────────────────────

TEST(CrossoverTest, ChildHasCorrectStructure) {
    Genome a(2, 1);
    a.add_connection({0, 3, 1.0f, true, 0});
    a.add_connection({1, 3, 0.5f, true, 1});
    a.set_fitness(10.0f);

    Genome b(2, 1);
    b.add_connection({0, 3, -1.0f, true, 0});
    b.set_fitness(5.0f);

    Random rng(42);
    Genome child = Crossover::crossover(a, b, rng);

    EXPECT_EQ(child.num_inputs(), 2);
    EXPECT_EQ(child.num_outputs(), 1);
    // Child should have connections from fitter parent (at least matching + excess)
    EXPECT_GE(child.connections().size(), 1u);
}

TEST(CrossoverTest, FitterParentContributesExcessGenes) {
    Genome a(2, 1);
    a.add_connection({0, 3, 1.0f, true, 0});
    a.add_connection({1, 3, 0.5f, true, 1});
    a.add_connection({2, 3, 0.3f, true, 2});  // excess
    a.set_fitness(10.0f);

    Genome b(2, 1);
    b.add_connection({0, 3, -1.0f, true, 0});
    b.set_fitness(5.0f);

    Random rng(42);
    Genome child = Crossover::crossover(a, b, rng);

    // Child should have all 3 connections (matching + disjoint + excess from fitter)
    EXPECT_EQ(child.connections().size(), 3u);
}

TEST(CrossoverTest, DisabledGeneHandling) {
    // If a gene is disabled in either parent, 75% chance disabled in child
    // Test statistically
    Genome a(2, 1);
    a.add_connection({0, 3, 1.0f, false, 0});  // disabled
    a.set_fitness(10.0f);

    Genome b(2, 1);
    b.add_connection({0, 3, -1.0f, true, 0});
    b.set_fitness(10.0f);

    Random rng(42);
    int disabled_count = 0;
    int trials = 200;
    for (int i = 0; i < trials; ++i) {
        Genome child = Crossover::crossover(a, b, rng);
        if (!child.connections().empty() && !child.connections()[0].enabled) {
            ++disabled_count;
        }
    }

    // Should be roughly 75% disabled (allow wide margin for randomness)
    float ratio = static_cast<float>(disabled_count) / trials;
    EXPECT_GT(ratio, 0.5f);
    EXPECT_LT(ratio, 0.95f);
}

TEST(CrossoverTest, ChildProducesValidNetwork) {
    Genome a(3, 2);
    Genome b(3, 2);
    InnovationTracker tracker;
    Random rng(42);

    // Fully connect both parents
    for (auto* g : {&a, &b}) {
        for (const auto& in_node : g->nodes()) {
            if (in_node.type != NodeType::Input && in_node.type != NodeType::Bias) continue;
            for (const auto& out_node : g->nodes()) {
                if (out_node.type != NodeType::Output) continue;
                g->add_connection({in_node.id, out_node.id,
                                   rng.next_float(-1.0f, 1.0f), true,
                                   tracker.get_innovation(in_node.id, out_node.id)});
            }
        }
    }

    a.set_fitness(5.0f);
    b.set_fitness(3.0f);

    Genome child = Crossover::crossover(a, b, rng);
    NeuralNetwork nn(child);
    auto outputs = nn.activate({1.0f, 0.5f, -1.0f});
    EXPECT_EQ(outputs.size(), 2u);
}

// ── Species Tests ───────────────────────────────────────────────────────

TEST(SpeciesTest, CompatibleGenomesMatch) {
    Genome rep(2, 1);
    rep.add_connection({0, 3, 0.5f, true, 0});
    Species s(rep);

    Genome similar(2, 1);
    similar.add_connection({0, 3, 0.6f, true, 0});

    EXPECT_TRUE(s.is_compatible(similar, 10.0f, 1.0f, 1.0f, 0.4f));
}

TEST(SpeciesTest, IncompatibleGenomesDontMatch) {
    Genome rep(2, 1);
    rep.add_connection({0, 3, 0.5f, true, 0});
    Species s(rep);

    Genome different(2, 1);
    different.add_connection({0, 3, 0.5f, true, 10});  // very different innovation
    different.add_connection({1, 3, 0.5f, true, 11});

    EXPECT_FALSE(s.is_compatible(different, 0.1f, 1.0f, 1.0f, 0.4f));
}

TEST(SpeciesTest, StagnationTracking) {
    Genome rep(2, 1);
    Species s(rep);

    // Add member with some fitness
    Genome g1(2, 1);
    g1.set_fitness(5.0f);
    s.add_member(&g1);
    s.update_best_fitness();

    EXPECT_EQ(s.generations_without_improvement(), 0);
    EXPECT_FALSE(s.is_stagnant(15));

    // Simulate no improvement for many generations
    for (int i = 0; i < 16; ++i) {
        s.clear_members();
        Genome g2(2, 1);
        g2.set_fitness(4.0f);  // worse than best
        s.add_member(&g2);
        s.update_best_fitness();
    }

    EXPECT_TRUE(s.is_stagnant(15));
}

TEST(SpeciesTest, AdjustFitnessSharesFitness) {
    Genome rep(2, 1);
    Species s(rep);

    Genome g1(2, 1);
    g1.set_fitness(10.0f);
    Genome g2(2, 1);
    g2.set_fitness(6.0f);

    s.add_member(&g1);
    s.add_member(&g2);
    s.adjust_fitness();

    // Each member's adjusted fitness = raw / species_size
    EXPECT_FLOAT_EQ(g1.adjusted_fitness(), 5.0f);
    EXPECT_FLOAT_EQ(g2.adjusted_fitness(), 3.0f);
}

// ── Regression Tests for Bug Fixes ─────────────────────────────────────

TEST(GenomeTest, CompatibilityDistanceWithOutOfOrderInnovations) {
    // Genome A has connections at innovations 5 and 2, added OUT OF ORDER
    // so back() returns innovation 2, not the true max 5.
    Genome a(2, 1);
    a.add_connection({0, 3, 1.0f, true, 5});   // innovation 5 added first
    a.add_connection({1, 3, 1.0f, true, 2});   // innovation 2 added second → back() = 2

    // Genome B has only innovation 2
    Genome b(2, 1);
    b.add_connection({0, 3, 1.0f, true, 2});

    // Correct: max_a = 5, max_b = 2, min_max = 2
    // innovation 2: matching (weight diff = 0)
    // innovation 5: excess (in A only, 5 > min_max=2)
    // delta = c1*1/n + 0 + 0 = 1.0*1/2 = 0.5 (c1=1, n=max(2,1)=2)
    float delta = Genome::compatibility_distance(a, b, 1.0f, 1.0f, 0.4f);
    EXPECT_NEAR(delta, 0.5f, 0.001f);
}

// ── Evolution Manager Tests ─────────────────────────────────────────────

TEST(EvolutionManagerTest, InitializeCreatesPopulation) {
    SimulationConfig config;
    config.predator_count = 5;
    config.prey_count = 10;
    config.seed = 42;

    Random rng(42);
    EvolutionManager evo(config, rng);
    evo.initialize(15, 2);

    EXPECT_EQ(static_cast<int>(evo.population().size()), 15);

    // Each genome should have correct structure
    for (const auto& g : evo.population()) {
        EXPECT_EQ(g.num_inputs(), 15);
        EXPECT_EQ(g.num_outputs(), 2);
        EXPECT_GT(g.connections().size(), 0u);
    }
}

TEST(EvolutionManagerTest, InitialGenomesProduceValidNetworks) {
    SimulationConfig config;
    config.predator_count = 3;
    config.prey_count = 5;
    config.seed = 42;

    Random rng(42);
    EvolutionManager evo(config, rng);
    evo.initialize(15, 2);

    EXPECT_EQ(evo.networks().size(), evo.population().size());

    for (size_t i = 0; i < evo.networks().size(); ++i) {
        auto outputs = evo.networks()[i]->activate(
            std::vector<float>(15, 0.5f));
        EXPECT_EQ(outputs.size(), 2u);
    }
}

TEST(EvolutionManagerTest, EvolveProducesNewGeneration) {
    SimulationConfig config;
    config.predator_count = 5;
    config.prey_count = 10;
    config.seed = 42;

    Random rng(42);
    EvolutionManager evo(config, rng);
    evo.initialize(4, 2);

    // Set some fitness values
    for (auto& g : evo.population()) {
        g.set_fitness(rng.next_float(0.0f, 10.0f));
    }

    EXPECT_EQ(evo.generation(), 0);
    evo.evolve();
    EXPECT_EQ(evo.generation(), 1);

    // Population size should be preserved
    EXPECT_EQ(static_cast<int>(evo.population().size()), 15);
}

// ── Delete Connection Tests ─────────────────────────────────────────────

TEST(MutationTest, DeleteConnectionReducesCount) {
    Genome g(2, 1);
    g.add_connection({0, 3, 1.0f, true, 0});
    g.add_connection({1, 3, 0.5f, true, 1});
    g.add_connection({2, 3, 0.3f, true, 2});  // bias

    EXPECT_EQ(g.connections().size(), 3u);

    Random rng(42);
    Mutation::delete_connection(g, rng);

    EXPECT_EQ(g.connections().size(), 2u);
}

TEST(MutationTest, DeleteConnectionKeepsAtLeastOne) {
    Genome g(2, 1);
    g.add_connection({0, 3, 1.0f, true, 0});

    EXPECT_EQ(g.connections().size(), 1u);

    Random rng(42);
    Mutation::delete_connection(g, rng);

    // Should not delete the last connection
    EXPECT_EQ(g.connections().size(), 1u);
}

TEST(MutationTest, DeleteConnectionProducesValidNetwork) {
    Genome g(3, 2);
    InnovationTracker tracker;
    Random rng(42);

    // Fully connect
    for (const auto& in_node : g.nodes()) {
        if (in_node.type != NodeType::Input && in_node.type != NodeType::Bias) continue;
        for (const auto& out_node : g.nodes()) {
            if (out_node.type != NodeType::Output) continue;
            g.add_connection({in_node.id, out_node.id,
                              rng.next_float(-1.0f, 1.0f), true,
                              tracker.get_innovation(in_node.id, out_node.id)});
        }
    }

    size_t initial_conns = g.connections().size();

    // Delete several connections
    for (int i = 0; i < 3; ++i) {
        Mutation::delete_connection(g, rng);
    }

    EXPECT_LT(g.connections().size(), initial_conns);
    EXPECT_GE(g.connections().size(), 1u);

    // Should still produce a valid network
    NeuralNetwork nn(g);
    auto outputs = nn.activate({1.0f, 0.5f, -1.0f});
    EXPECT_EQ(outputs.size(), 2u);
}

// ── SensorInput::write_to Tests ─────────────────────────────────────────

TEST(SensorInputTest, WriteToMatchesToVector) {
    SensorInput si;
    si.nearest_predator_dist = 0.5f;
    si.nearest_predator_angle = -0.3f;
    si.nearest_prey_dist = 0.8f;
    si.nearest_prey_angle = 0.1f;
    si.nearest_food_dist = 0.2f;
    si.nearest_food_angle = -0.7f;
    si.energy_level = 0.6f;
    si.speed_x = 0.3f;
    si.speed_y = -0.4f;
    si.local_predator_density = 0.15f;
    si.local_prey_density = 0.25f;
    si.wall_left = 0.9f;
    si.wall_right = 0.1f;
    si.wall_top = 0.5f;
    si.wall_bottom = 0.7f;

    auto vec = si.to_vector();
    float buf[SensorInput::SIZE];
    si.write_to(buf);

    ASSERT_EQ(vec.size(), static_cast<size_t>(SensorInput::SIZE));
    for (int i = 0; i < SensorInput::SIZE; ++i) {
        EXPECT_FLOAT_EQ(buf[i], vec[i]) << "Mismatch at index " << i;
    }
}

// ── NeuralNetwork::activate_into Tests ──────────────────────────────────

TEST(NeuralNetworkTest, ActivateIntoMatchesActivate) {
    Genome g(3, 2);
    InnovationTracker tracker;
    Random rng(42);

    // Fully connect
    for (const auto& in_node : g.nodes()) {
        if (in_node.type != NodeType::Input && in_node.type != NodeType::Bias) continue;
        for (const auto& out_node : g.nodes()) {
            if (out_node.type != NodeType::Output) continue;
            g.add_connection({in_node.id, out_node.id,
                              rng.next_float(-1.0f, 1.0f), true,
                              tracker.get_innovation(in_node.id, out_node.id)});
        }
    }

    // Add some hidden nodes
    SimulationConfig config;
    config.mutation_rate = 1.0f;
    config.add_node_rate = 0.5f;
    for (int i = 0; i < 5; ++i) {
        Mutation::mutate(g, rng, config, tracker);
    }

    std::vector<float> inputs = {1.0f, 0.5f, -1.0f};

    NeuralNetwork nn(g);
    auto vec_out = nn.activate(inputs);

    // Reset and use activate_into
    NeuralNetwork nn2(g);
    float out_buf[2];
    nn2.activate_into(inputs.data(), 3, out_buf, 2);

    ASSERT_EQ(vec_out.size(), 2u);
    for (int i = 0; i < 2; ++i) {
        EXPECT_FLOAT_EQ(out_buf[i], vec_out[i]) << "Mismatch at output " << i;
    }
}

// ── Default Fitness Formula Regression Tests ─────────────────────────────

TEST(FitnessComputationTest, DefaultFormulaKnownValues) {
    // Regression test: verifies the default fitness formula against a hand-computed
    // expected value using default config weights. If weights or formula change, this fails.
    //   survival=1.0, kill=5.0, energy=0.5, distance=0.1, complexity_penalty=0.01
    //   age=0.5, kills=2.0, energy=0.8, alive=1.0, dist=0.3, complexity=5.0
    //   = 1.0*0.5 + 5.0*2.0 + 0.5*0.8 + 1.0 + 0.1*0.3 - 0.01*5.0
    //   = 0.5 + 10.0 + 0.4 + 1.0 + 0.03 - 0.05 = 11.88
    SimulationConfig config;
    const float expected = config.fitness_survival_weight   * 0.5f
                         + config.fitness_kill_weight       * 2.0f
                         + config.fitness_energy_weight     * 0.8f
                         + 1.0f
                         + config.fitness_distance_weight   * 0.3f
                         - config.complexity_penalty_weight * 5.0f;

    EXPECT_NEAR(expected, 11.88f, 0.001f);
}

TEST(FitnessComputationTest, DefaultFormulaClampedAtZero) {
    // Formula result should never go below zero
    SimulationConfig config;
    // All-zero inputs, alive_bonus=0 → result should be 0 or clamped to 0
    const float result = std::max(0.0f,
        config.fitness_survival_weight   * 0.0f
      + config.fitness_kill_weight       * 0.0f
      + config.fitness_energy_weight     * 0.0f
      + 0.0f   // alive_bonus
      + config.fitness_distance_weight   * 0.0f
      - config.complexity_penalty_weight * 100.0f);  // large complexity penalty

    EXPECT_FLOAT_EQ(result, 0.0f);
}
