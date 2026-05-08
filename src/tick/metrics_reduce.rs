use serde::{Deserialize, Serialize};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MetricsSummaryReadback {
    pub tick: u32,
    pub predator_count: u32,
    pub prey_count: u32,
    pub predator_births: u32,
    pub prey_births: u32,
    pub predator_deaths: u32,
    pub prey_deaths: u32,
    pub predator_species: u32,
    pub prey_species: u32,
    pub avg_predator_complexity: f32,
    pub avg_prey_complexity: f32,
    pub avg_predator_energy: f32,
    pub avg_prey_energy: f32,
    pub max_predator_generation: u32,
    pub avg_predator_generation: f32,
    pub max_prey_generation: u32,
    pub avg_prey_generation: f32,
}
