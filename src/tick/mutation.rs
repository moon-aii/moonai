use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::config::SimulationConfig;
use crate::tick::genome::PopulationKind;

pub const PHASE3_MAX_CONNECTION_ATTEMPTS: u32 = 16;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuMutationConfig {
    pub mutation_rate: f32,
    pub weight_mutation_power: f32,
    pub add_node_rate: f32,
    pub add_connection_rate: f32,
    pub delete_connection_rate: f32,
    pub max_connection_attempts: u32,
}

impl GpuMutationConfig {
    pub fn for_simulation(config: &SimulationConfig) -> Result<Self> {
        for (name, value) in [
            ("mutation_rate", config.mutation_rate),
            ("add_node_rate", config.add_node_rate),
            ("add_connection_rate", config.add_connection_rate),
            ("delete_connection_rate", config.delete_connection_rate),
        ] {
            if !(0.0..=1.0).contains(&value) {
                bail!("{name} must be in [0, 1], got {value}");
            }
        }
        if config.weight_mutation_power <= 0.0 {
            bail!("weight_mutation_power must be positive for GPU mutation, got {}", config.weight_mutation_power);
        }

        Ok(Self {
            mutation_rate: config.mutation_rate,
            weight_mutation_power: config.weight_mutation_power,
            add_node_rate: config.add_node_rate,
            add_connection_rate: config.add_connection_rate,
            delete_connection_rate: config.delete_connection_rate,
            max_connection_attempts: PHASE3_MAX_CONNECTION_ATTEMPTS,
        })
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MutationSummaryReadback {
    pub population_kind: PopulationKind,
    pub agents_mutated: u32,
    pub weight_perturbations: u32,
    pub added_connections: u32,
    pub added_nodes: u32,
    pub deleted_connections: u32,
}
