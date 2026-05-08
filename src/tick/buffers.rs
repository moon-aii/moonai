use serde::{Deserialize, Serialize};

use crate::tick::compiled::DeviceCompiledNetworkBuffers;
use crate::tick::genome::{DeviceGenomeBuffers, PopulationKind};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DevicePopulationBuffers {
    pub pos_x: *mut f32,
    pub pos_y: *mut f32,
    pub vel_x: *mut f32,
    pub vel_y: *mut f32,
    pub energy: *mut f32,
    pub age: *mut f32,
    pub alive: *mut u8,
    pub species_id: *mut u32,
    pub entity_id: *mut u32,
    pub generation: *mut u32,
    pub rng_state: *mut u64,
    pub sensor_inputs: *mut f32,
    pub genome: DeviceGenomeBuffers,
    pub compiled: DeviceCompiledNetworkBuffers,
    pub capacity: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PredatorBuffer {
    pub population: DevicePopulationBuffers,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PreyBuffer {
    pub population: DevicePopulationBuffers,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FoodBuffer {
    pub pos_x: *mut f32,
    pub pos_y: *mut f32,
    pub active: *mut u8,
    pub capacity: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UiStatsReadback {
    pub tick: u32,
    pub predator_count: u32,
    pub prey_count: u32,
    pub predator_births: u32,
    pub prey_births: u32,
    pub predator_deaths: u32,
    pub prey_deaths: u32,
    pub kills: u32,
    pub food_eaten: u32,
    pub avg_predator_energy: f32,
    pub avg_prey_energy: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PopulationSummaryReadback {
    pub population_kind: PopulationKind,
    pub live_count: u32,
    pub capacity: u32,
    pub next_entity_id: u32,
    pub innovation_counter: u32,
    pub next_node_id: u32,
    pub avg_energy: f32,
    pub avg_connections: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RenderSnapshotHeader {
    pub tick: u32,
    pub total_predators: u32,
    pub total_prey: u32,
    pub total_food: u32,
    pub returned_predators: u32,
    pub returned_prey: u32,
    pub returned_food: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RenderAgentReadback {
    pub population_kind: PopulationKind,
    pub slot: u32,
    pub entity_id: u32,
    pub species_id: u32,
    pub generation: u32,
    pub age: f32,
    pub pos_x: f32,
    pub pos_y: f32,
    pub dir_x: f32,
    pub dir_y: f32,
    pub energy: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RenderFoodReadback {
    pub slot: u32,
    pub active: u8,
    pub reserved0: u8,
    pub reserved1: u16,
    pub pos_x: f32,
    pub pos_y: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpatialGridReadback {
    pub grid_cols: u32,
    pub grid_rows: u32,
    pub cell_count: u32,
    pub predator_entries: u32,
    pub prey_entries: u32,
    pub food_entries: u32,
    pub cell_size: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderSnapshotReadback {
    pub header: RenderSnapshotHeader,
    pub predators: Vec<RenderAgentReadback>,
    pub prey: Vec<RenderAgentReadback>,
    pub food: Vec<RenderFoodReadback>,
}
