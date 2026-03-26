#include "core/config.hpp"
#include "core/random.hpp"
#include "evolution/crossover.hpp"
#include "evolution/genome.hpp"
#include "evolution/mutation.hpp"
#include "evolution/neural_network.hpp"
#include "evolution/species.hpp"
#include "simulation/entity.hpp"
#include <gtest/gtest.h>

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
  tracker.reset_mutation_window();
  std::uint32_t i2 = tracker.get_innovation(0, 3);
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

  EXPECT_GE(tracker.innovation_count(), 6u);
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

  bool changed = (g.connections()[0].weight != original_w0 ||
                  g.connections()[1].weight != original_w1);
  EXPECT_TRUE(changed);
}

TEST(MutationTest, WeightsAreClamped) {
  Genome g(1, 1);
  g.add_connection({0, 2, 7.5f, true, 0});

  Random rng(42);
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

  for (const auto &conn : g.connections()) {
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
  EXPECT_FALSE(g.connections()[0].enabled);
  EXPECT_EQ(g.connections().size(), 3u);
}

TEST(MutationTest, MutatedGenomeProducesValidNetwork) {
  Genome g(3, 2);
  InnovationTracker tracker;
  Random rng(42);

  for (const auto &in_node : g.nodes()) {
    if (in_node.type != NodeType::Input && in_node.type != NodeType::Bias)
      continue;
    for (const auto &out_node : g.nodes()) {
      if (out_node.type != NodeType::Output)
        continue;
      g.add_connection({in_node.id, out_node.id, rng.next_float(-1.0f, 1.0f),
                        true, tracker.get_innovation(in_node.id, out_node.id)});
    }
  }

  SimulationConfig config;
  config.mutation_rate = 1.0f;
  config.add_connection_rate = 0.5f;
  config.add_node_rate = 0.3f;

  for (int i = 0; i < 20; ++i) {
    Mutation::mutate(g, rng, config, tracker);
  }

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
  EXPECT_GE(child.connections().size(), 1u);
}

TEST(CrossoverTest, FitterParentContributesExcessGenes) {
  Genome a(2, 1);
  a.add_connection({0, 3, 1.0f, true, 0});
  a.add_connection({1, 3, 0.5f, true, 1});
  a.add_connection({2, 3, 0.3f, true, 2});
  a.set_fitness(10.0f);

  Genome b(2, 1);
  b.add_connection({0, 3, -1.0f, true, 0});
  b.set_fitness(5.0f);

  Random rng(42);
  Genome child = Crossover::crossover(a, b, rng);

  EXPECT_EQ(child.connections().size(), 3u);
}

TEST(CrossoverTest, DisabledGeneHandling) {
  Genome a(2, 1);
  a.add_connection({0, 3, 1.0f, false, 0});
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

  float ratio = static_cast<float>(disabled_count) / trials;
  EXPECT_GT(ratio, 0.5f);
  EXPECT_LT(ratio, 0.95f);
}

TEST(CrossoverTest, ChildProducesValidNetwork) {
  Genome a(3, 2);
  Genome b(3, 2);
  InnovationTracker tracker;
  Random rng(42);

  for (auto *g : {&a, &b}) {
    for (const auto &in_node : g->nodes()) {
      if (in_node.type != NodeType::Input && in_node.type != NodeType::Bias)
        continue;
      for (const auto &out_node : g->nodes()) {
        if (out_node.type != NodeType::Output)
          continue;
        g->add_connection({in_node.id, out_node.id, rng.next_float(-1.0f, 1.0f),
                           true,
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
  different.add_connection({0, 3, 0.5f, true, 10});
  different.add_connection({1, 3, 0.5f, true, 11});

  EXPECT_FALSE(s.is_compatible(different, 0.1f, 1.0f, 1.0f, 0.4f));
}

TEST(SpeciesTest, SummaryTracking) {
  Genome rep(2, 1);
  Species s(rep);

  Genome g1(2, 1);
  g1.set_fitness(5.0f);
  s.add_member(Entity{1, 1}, g1);
  s.refresh_summary();

  EXPECT_FLOAT_EQ(s.average_fitness(), 5.0f);
  EXPECT_FLOAT_EQ(s.best_fitness_ever(), 5.0f);
  EXPECT_EQ(s.members().size(), 1u);
}

TEST(SpeciesTest, AverageFitnessUsesMembers) {
  Genome rep(2, 1);
  Species s(rep);

  Genome g1(2, 1);
  g1.set_fitness(10.0f);
  Genome g2(2, 1);
  g2.set_fitness(6.0f);

  s.add_member(Entity{1, 1}, g1);
  s.add_member(Entity{2, 1}, g2);
  s.refresh_summary();

  EXPECT_FLOAT_EQ(s.average_fitness(), 8.0f);
  EXPECT_FLOAT_EQ(s.best_fitness_ever(), 10.0f);
}

// ── Regression Tests for Bug Fixes ─────────────────────────────────────

TEST(GenomeTest, CompatibilityDistanceWithOutOfOrderInnovations) {
  Genome a(2, 1);
  a.add_connection({0, 3, 1.0f, true, 5});
  a.add_connection({1, 3, 1.0f, true, 2});

  Genome b(2, 1);
  b.add_connection({0, 3, 1.0f, true, 2});

  float delta = Genome::compatibility_distance(a, b, 1.0f, 1.0f, 0.4f);
  EXPECT_NEAR(delta, 0.5f, 0.001f);
}

// ── Delete Connection Tests ─────────────────────────────────────────────

TEST(MutationTest, DeleteConnectionReducesCount) {
  Genome g(2, 1);
  g.add_connection({0, 3, 1.0f, true, 0});
  g.add_connection({1, 3, 0.5f, true, 1});
  g.add_connection({2, 3, 0.3f, true, 2});

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

  EXPECT_EQ(g.connections().size(), 1u);
}

TEST(MutationTest, DeleteConnectionProducesValidNetwork) {
  Genome g(3, 2);
  InnovationTracker tracker;
  Random rng(42);

  for (const auto &in_node : g.nodes()) {
    if (in_node.type != NodeType::Input && in_node.type != NodeType::Bias)
      continue;
    for (const auto &out_node : g.nodes()) {
      if (out_node.type != NodeType::Output)
        continue;
      g.add_connection({in_node.id, out_node.id, rng.next_float(-1.0f, 1.0f),
                        true, tracker.get_innovation(in_node.id, out_node.id)});
    }
  }

  size_t initial_conns = g.connections().size();

  for (int i = 0; i < 3; ++i) {
    Mutation::delete_connection(g, rng);
  }

  EXPECT_LT(g.connections().size(), initial_conns);
  EXPECT_GE(g.connections().size(), 1u);

  NeuralNetwork nn(g);
  auto outputs = nn.activate({1.0f, 0.5f, -1.0f});
  EXPECT_EQ(outputs.size(), 2u);
}

// ── NeuralNetwork::activate_into Tests ──────────────────────────────────

TEST(NeuralNetworkTest, ActivateIntoMatchesActivate) {
  Genome g(3, 2);
  InnovationTracker tracker;
  Random rng(42);

  for (const auto &in_node : g.nodes()) {
    if (in_node.type != NodeType::Input && in_node.type != NodeType::Bias)
      continue;
    for (const auto &out_node : g.nodes()) {
      if (out_node.type != NodeType::Output)
        continue;
      g.add_connection({in_node.id, out_node.id, rng.next_float(-1.0f, 1.0f),
                        true, tracker.get_innovation(in_node.id, out_node.id)});
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

  NeuralNetwork nn2(g);
  float out_buf[2];
  nn2.activate_into(inputs.data(), 3, out_buf, 2);

  ASSERT_EQ(vec_out.size(), 2u);
  for (int i = 0; i < 2; ++i) {
    EXPECT_FLOAT_EQ(out_buf[i], vec_out[i]) << "Mismatch at output " << i;
  }
}

// ── Genome Tests ─────────────────────────────────────────────────────────

TEST(GenomeTest, ConstructorCreatesInputAndOutputNodes) {
  Genome g(3, 2);

  EXPECT_EQ(g.nodes().size(), 6u);
  EXPECT_EQ(g.num_inputs(), 3);
  EXPECT_EQ(g.num_outputs(), 2);
}

TEST(GenomeTest, AddConnection) {
  Genome g(2, 1);
  g.add_connection({0, 3, 0.5f, true, 0});

  EXPECT_EQ(g.connections().size(), 1u);
  EXPECT_FLOAT_EQ(g.connections()[0].weight, 0.5f);
}

TEST(GenomeTest, HasConnection) {
  Genome g(2, 1);
  g.add_connection({0, 3, 0.5f, true, 0});

  EXPECT_TRUE(g.has_connection(0, 3));
  EXPECT_FALSE(g.has_connection(1, 3));
  EXPECT_FALSE(g.has_connection(3, 0));
}

TEST(GenomeTest, HasNode) {
  Genome g(2, 1);
  EXPECT_TRUE(g.has_node(0));
  EXPECT_TRUE(g.has_node(1));
  EXPECT_TRUE(g.has_node(2));
  EXPECT_TRUE(g.has_node(3));
  EXPECT_FALSE(g.has_node(4));
}

TEST(GenomeTest, MaxNodeId) {
  Genome g(2, 1);
  EXPECT_EQ(g.max_node_id(), 3u);

  g.add_node({10, NodeType::Hidden});
  EXPECT_EQ(g.max_node_id(), 10u);
}

TEST(GenomeTest, Complexity) {
  Genome g(2, 1);
  EXPECT_EQ(g.complexity(), 4);

  g.add_connection({0, 3, 0.5f, true, 0});
  EXPECT_EQ(g.complexity(), 5);
}

TEST(GenomeTest, CompatibilityDistanceSameGenome) {
  Genome g(2, 1);
  g.add_connection({0, 3, 0.5f, true, 0});

  float dist = Genome::compatibility_distance(g, g, 1.0f, 1.0f, 0.4f);
  EXPECT_FLOAT_EQ(dist, 0.0f);
}

TEST(GenomeTest, CompatibilityDistanceDifferentWeights) {
  Genome a(2, 1);
  a.add_connection({0, 3, 1.0f, true, 0});

  Genome b(2, 1);
  b.add_connection({0, 3, 0.0f, true, 0});

  float dist = Genome::compatibility_distance(a, b, 0.0f, 0.0f, 1.0f);
  EXPECT_FLOAT_EQ(dist, 1.0f);
}

TEST(GenomeTest, CompatibilityDistanceWithExcess) {
  Genome a(2, 1);
  a.add_connection({0, 3, 0.5f, true, 0});
  a.add_connection({1, 3, 0.5f, true, 1});
  a.add_connection({2, 3, 0.5f, true, 2});

  Genome b(2, 1);
  b.add_connection({0, 3, 0.5f, true, 0});

  float dist = Genome::compatibility_distance(a, b, 1.0f, 0.0f, 0.0f);
  EXPECT_GT(dist, 0.0f);
}

TEST(GenomeTest, FitnessDefaultsToZero) {
  Genome g(2, 1);
  EXPECT_FLOAT_EQ(g.fitness(), 0.0f);
}

TEST(GenomeTest, AdjustedFitness) {
  Genome g(2, 1);
  g.set_adjusted_fitness(0.5f);
  EXPECT_FLOAT_EQ(g.adjusted_fitness(), 0.5f);
}

TEST(GenomeTest, JsonRoundTrip) {
  Genome g(3, 2);
  g.add_connection({0, 4, 0.5f, true, 0});
  g.add_connection({1, 5, -0.3f, false, 1});
  g.add_node({10, NodeType::Hidden});
  g.set_fitness(42.0f);

  std::string json = g.to_json();
  Genome restored = Genome::from_json(json);

  EXPECT_EQ(restored.num_inputs(), 3);
  EXPECT_EQ(restored.num_outputs(), 2);
  EXPECT_EQ(restored.nodes().size(), g.nodes().size());
  EXPECT_EQ(restored.connections().size(), 2u);
  EXPECT_FLOAT_EQ(restored.fitness(), 42.0f);
  EXPECT_FLOAT_EQ(restored.connections()[0].weight, 0.5f);
  EXPECT_FALSE(restored.connections()[1].enabled);
}

// ── Neural Network Tests ─────────────────────────────────────────────────

TEST(NeuralNetworkTest, ActivateReturnsCorrectOutputCount) {
  Genome g(3, 2);

  std::uint32_t innov = 0;
  for (int i = 0; i < 3; ++i) {
    for (int o = 4; o <= 5; ++o) {
      g.add_connection({static_cast<std::uint32_t>(i),
                        static_cast<std::uint32_t>(o), 0.5f, true, innov++});
    }
  }

  NeuralNetwork nn(g);
  auto outputs = nn.activate({1.0f, 0.5f, -1.0f});

  EXPECT_EQ(outputs.size(), 2u);
}

TEST(NeuralNetworkTest, OutputsAreBounded) {
  Genome g(2, 1);
  g.add_connection({0, 3, 100.0f, true, 0});

  NeuralNetwork nn(g);
  auto outputs = nn.activate({1.0f, 0.0f});

  EXPECT_GE(outputs[0], 0.0f);
  EXPECT_LE(outputs[0], 1.0f);
}

TEST(NeuralNetworkTest, BiasNodeIsActive) {
  Genome g(1, 1);
  g.add_connection({1, 2, 1.0f, true, 0});

  NeuralNetwork nn(g);
  auto outputs = nn.activate({0.0f});

  EXPECT_GT(outputs[0], 0.9f);
}

TEST(NeuralNetworkTest, DisabledConnectionsIgnored) {
  Genome g(1, 1);
  g.add_connection({0, 2, 100.0f, false, 0});
  g.add_connection({1, 2, 0.0f, true, 1});

  NeuralNetwork nn(g);
  auto outputs = nn.activate({1.0f});

  EXPECT_NEAR(outputs[0], 0.5f, 0.01f);
}

TEST(NeuralNetworkTest, HiddenNodeTopologicalSort) {
  Genome g(1, 1);
  g.add_node({3, NodeType::Hidden});

  g.add_connection({0, 3, 1.0f, true, 0});
  g.add_connection({3, 2, 1.0f, true, 1});

  NeuralNetwork nn(g);
  auto outputs = nn.activate({1.0f});

  EXPECT_GT(outputs[0], 0.9f);
  EXPECT_LE(outputs[0], 1.0f);
}

TEST(NeuralNetworkTest, MultipleHiddenLayers) {
  Genome g(1, 1);
  g.add_node({3, NodeType::Hidden});
  g.add_node({4, NodeType::Hidden});

  g.add_connection({0, 3, 1.0f, true, 0});
  g.add_connection({3, 4, 1.0f, true, 1});
  g.add_connection({4, 2, 1.0f, true, 2});

  NeuralNetwork nn(g);
  auto outputs = nn.activate({1.0f});

  EXPECT_GE(outputs[0], 0.0f);
  EXPECT_LE(outputs[0], 1.0f);
}

TEST(NeuralNetworkTest, ZeroInputsProduceDefaultOutput) {
  Genome g(2, 1);
  g.add_connection({0, 3, 1.0f, true, 0});
  g.add_connection({1, 3, 1.0f, true, 1});

  NeuralNetwork nn(g);
  auto outputs = nn.activate({0.0f, 0.0f});

  EXPECT_NEAR(outputs[0], 0.5f, 0.01f);
}

// ── Default Fitness Formula Regression Tests ─────────────────────────────

TEST(FitnessComputationTest, DefaultFormulaClampedAtZero) {
  SimulationConfig config;
  const float result =
      std::max(0.0f, config.fitness_survival_weight * 0.0f +
                         config.fitness_kill_weight * 0.0f +
                         config.fitness_energy_weight * 0.0f + 0.0f +
                         config.fitness_distance_weight * 0.0f -
                         config.complexity_penalty_weight * 100.0f);

  EXPECT_FLOAT_EQ(result, 0.0f);
}
