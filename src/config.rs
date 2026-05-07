use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    #[serde(default = "grid_size_default")]
    pub grid_size: i32,
    #[serde(default = "predator_count_default")]
    pub predator_count: i32,
    #[serde(default = "prey_count_default")]
    pub prey_count: i32,
    #[serde(default = "food_count_default")]
    pub food_count: i32,
    #[serde(default = "predator_speed_default")]
    pub predator_speed: f32,
    #[serde(default = "prey_speed_default")]
    pub prey_speed: f32,
    #[serde(default = "vision_range_default")]
    pub vision_range: f32,
    #[serde(default = "interaction_range_default")]
    pub interaction_range: f32,
    #[serde(default = "mate_range_default")]
    pub mate_range: f32,
    #[serde(default = "food_respawn_rate_default")]
    pub food_respawn_rate: f32,
    #[serde(default = "energy_drain_per_tick_default")]
    pub energy_drain_per_tick: f32,
    #[serde(default = "energy_gain_from_kill_default")]
    pub energy_gain_from_kill: f32,
    #[serde(default = "energy_gain_from_food_default")]
    pub energy_gain_from_food: f32,
    #[serde(default = "initial_energy_default")]
    pub initial_energy: f32,
    #[serde(default = "max_energy_default")]
    pub max_energy: f32,
    #[serde(default = "reproduction_energy_threshold_default")]
    pub reproduction_energy_threshold: f32,
    #[serde(default = "reproduction_energy_cost_default")]
    pub reproduction_energy_cost: f32,
    #[serde(default = "offspring_initial_energy_default")]
    pub offspring_initial_energy: f32,
    #[serde(default = "max_age_default")]
    pub max_age: i32,
    #[serde(default = "mutation_rate_default")]
    pub mutation_rate: f32,
    #[serde(default = "weight_mutation_power_default")]
    pub weight_mutation_power: f32,
    #[serde(default = "add_node_rate_default")]
    pub add_node_rate: f32,
    #[serde(default = "add_connection_rate_default")]
    pub add_connection_rate: f32,
    #[serde(default = "delete_connection_rate_default")]
    pub delete_connection_rate: f32,
    #[serde(default = "max_hidden_nodes_default")]
    pub max_hidden_nodes: i32,
    #[serde(default = "max_ticks_default")]
    pub max_ticks: i32,
    #[serde(default = "compatibility_threshold_default")]
    pub compatibility_threshold: f32,
    #[serde(default = "compatibility_min_normalization_default")]
    pub compatibility_min_normalization: f32,
    #[serde(default = "c1_excess_default")]
    pub c1_excess: f32,
    #[serde(default = "c2_disjoint_default")]
    pub c2_disjoint: f32,
    #[serde(default = "c3_weight_default")]
    pub c3_weight: f32,
    #[serde(default = "seed_default")]
    pub seed: i32,
    #[serde(default = "output_dir_default")]
    pub output_dir: String,
    #[serde(default = "report_interval_ticks_default")]
    pub report_interval_ticks: i32,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            grid_size: 3600,
            predator_count: 24000,
            prey_count: 96000,
            food_count: 240000,
            predator_speed: 1.0,
            prey_speed: 1.006,
            vision_range: 12.0,
            interaction_range: 1.0,
            mate_range: 6.0,
            food_respawn_rate: 0.006,
            energy_drain_per_tick: 0.001,
            energy_gain_from_kill: 0.24,
            energy_gain_from_food: 0.24,
            initial_energy: 0.36,
            max_energy: 2.0,
            reproduction_energy_threshold: 1.0,
            reproduction_energy_cost: 0.18,
            offspring_initial_energy: 0.36,
            max_age: 10000,
            mutation_rate: 0.30,
            weight_mutation_power: 0.30,
            add_node_rate: 0.12,
            add_connection_rate: 0.60,
            delete_connection_rate: 0.00000006,
            max_hidden_nodes: 1200,
            max_ticks: 0,
            compatibility_threshold: 60.0,
            compatibility_min_normalization: 240.0,
            c1_excess: 1.0,
            c2_disjoint: 1.0,
            c3_weight: 0.4,
            seed: 67,
            output_dir: "output/experiments".to_owned(),
            report_interval_ticks: 1000,
        }
    }
}

const fn grid_size_default() -> i32 {
    3600
}
const fn predator_count_default() -> i32 {
    24000
}
const fn prey_count_default() -> i32 {
    96000
}
const fn food_count_default() -> i32 {
    240000
}
const fn predator_speed_default() -> f32 {
    1.0
}
const fn prey_speed_default() -> f32 {
    1.006
}
const fn vision_range_default() -> f32 {
    12.0
}
const fn interaction_range_default() -> f32 {
    1.0
}
const fn mate_range_default() -> f32 {
    6.0
}
const fn food_respawn_rate_default() -> f32 {
    0.006
}
const fn energy_drain_per_tick_default() -> f32 {
    0.001
}
const fn energy_gain_from_kill_default() -> f32 {
    0.24
}
const fn energy_gain_from_food_default() -> f32 {
    0.24
}
const fn initial_energy_default() -> f32 {
    0.36
}
const fn max_energy_default() -> f32 {
    2.0
}
const fn reproduction_energy_threshold_default() -> f32 {
    1.0
}
const fn reproduction_energy_cost_default() -> f32 {
    0.18
}
const fn offspring_initial_energy_default() -> f32 {
    0.36
}
const fn max_age_default() -> i32 {
    10000
}
const fn mutation_rate_default() -> f32 {
    0.30
}
const fn weight_mutation_power_default() -> f32 {
    0.30
}
const fn add_node_rate_default() -> f32 {
    0.12
}
const fn add_connection_rate_default() -> f32 {
    0.60
}
const fn delete_connection_rate_default() -> f32 {
    0.00000006
}
const fn max_hidden_nodes_default() -> i32 {
    1200
}
const fn max_ticks_default() -> i32 {
    0
}
const fn compatibility_threshold_default() -> f32 {
    60.0
}
const fn compatibility_min_normalization_default() -> f32 {
    240.0
}
const fn c1_excess_default() -> f32 {
    1.0
}
const fn c2_disjoint_default() -> f32 {
    1.0
}
const fn c3_weight_default() -> f32 {
    0.4
}
const fn seed_default() -> i32 {
    67
}
fn output_dir_default() -> String {
    "output/experiments".to_owned()
}
const fn report_interval_ticks_default() -> i32 {
    1000
}
