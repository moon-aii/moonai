mod bootstrap;
mod combat;
mod crossover;
mod dynamics;
mod food;
mod helpers;
mod metrics;
mod mutation;
mod network_compilation;
mod render;
mod reproduction;
mod sensing;
mod spatial_grid;

use self::helpers::*;

use crate::experiment::SimulationConfig;
use crate::profile_scope;
use anyhow::{Context as _, Result, bail};
use serde::{Deserialize, Serialize};
use std::mem::MaybeUninit;

pub const MAX_SPECIES_SUMMARIES: u32 = 64;
pub const SENSOR_COUNT: u32 = 35;
pub const OUTPUT_COUNT: u32 = 2;
const PHASE3_CONNECTION_GROWTH_BUDGET_CAP: u32 = 16;

fn checked_usize_product(lhs: usize, rhs: usize, context: &'static str) -> Result<usize> {
    lhs.checked_mul(rhs).context(context)
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct DeviceGenomeBuffers {
    connection_from: *mut i32,
    connection_to: *mut i32,
    connection_weight: *mut f32,
    connection_innovation: *mut u32,
    connection_enabled: *mut u8,
    node_types: *mut u8,
    num_connections: *mut u16,
    num_nodes: *mut u16,
    connection_stride: u32,
    node_stride: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct DeviceCompiledNetworkBuffers {
    eval_order: *mut u16,
    connection_offsets: *mut u32,
    output_indices: *mut u16,
    connection_sources: *mut u16,
    connection_weights: *mut f32,
    node_counts: *mut u16,
    eval_counts: *mut u16,
    connection_counts: *mut u16,
    node_stride: u32,
    connection_stride: u32,
    output_stride: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct DevicePopulationBuffers {
    pos_x: *mut f32,
    pos_y: *mut f32,
    vel_x: *mut f32,
    vel_y: *mut f32,
    energy: *mut f32,
    age: *mut f32,
    alive: *mut u8,
    species_id: *mut u32,
    entity_id: *mut u32,
    generation: *mut u32,
    rng_state: *mut u64,
    sensor_inputs: *mut f32,
    genome: DeviceGenomeBuffers,
    compiled: DeviceCompiledNetworkBuffers,
    capacity: u32,
}

impl DevicePopulationBuffers {
    fn allocate(
        &mut self,
        capacity: u32,
        num_inputs: u32,
        node_stride: u32,
        connection_stride: u32,
        output_stride: u32,
    ) -> Result<()> {
        self.capacity = capacity;
        self.genome.connection_stride = connection_stride;
        self.genome.node_stride = node_stride;
        self.compiled.node_stride = node_stride;
        self.compiled.connection_stride = connection_stride;
        self.compiled.output_stride = output_stride;

        let result = (|| -> Result<()> {
            let capacity = capacity as usize;
            let num_inputs = num_inputs as usize;
            let node_stride = node_stride as usize;
            let connection_stride = connection_stride as usize;
            let output_stride = output_stride as usize;
            let offset_stride = node_stride.checked_add(1).context("population compiled offset stride overflowed")?;
            let sensor_entries =
                checked_usize_product(capacity, num_inputs, "population sensor buffer size overflowed")?;
            let connection_entries =
                checked_usize_product(capacity, connection_stride, "population connection buffer size overflowed")?;
            let node_entries = checked_usize_product(capacity, node_stride, "population node buffer size overflowed")?;
            let offset_entries =
                checked_usize_product(capacity, offset_stride, "population compiled offset buffer size overflowed")?;
            let output_entries =
                checked_usize_product(capacity, output_stride, "population output buffer size overflowed")?;

            cuda_malloc(&mut self.pos_x, capacity)?;
            cuda_malloc(&mut self.pos_y, capacity)?;
            cuda_malloc(&mut self.vel_x, capacity)?;
            cuda_malloc(&mut self.vel_y, capacity)?;
            cuda_malloc(&mut self.energy, capacity)?;
            cuda_malloc(&mut self.age, capacity)?;
            cuda_malloc(&mut self.alive, capacity)?;
            cuda_malloc(&mut self.species_id, capacity)?;
            cuda_malloc(&mut self.entity_id, capacity)?;
            cuda_malloc(&mut self.generation, capacity)?;
            cuda_malloc(&mut self.rng_state, capacity)?;
            cuda_malloc(&mut self.sensor_inputs, sensor_entries)?;
            cuda_malloc(&mut self.genome.connection_from, connection_entries)?;
            cuda_malloc(&mut self.genome.connection_to, connection_entries)?;
            cuda_malloc(&mut self.genome.connection_weight, connection_entries)?;
            cuda_malloc(&mut self.genome.connection_innovation, connection_entries)?;
            cuda_malloc(&mut self.genome.connection_enabled, connection_entries)?;
            cuda_malloc(&mut self.genome.node_types, node_entries)?;
            cuda_malloc(&mut self.genome.num_connections, capacity)?;
            cuda_malloc(&mut self.genome.num_nodes, capacity)?;
            cuda_malloc(&mut self.compiled.eval_order, node_entries)?;
            cuda_malloc(&mut self.compiled.connection_offsets, offset_entries)?;
            cuda_malloc(&mut self.compiled.output_indices, output_entries)?;
            cuda_malloc(&mut self.compiled.connection_sources, connection_entries)?;
            cuda_malloc(&mut self.compiled.connection_weights, connection_entries)?;
            cuda_malloc(&mut self.compiled.node_counts, capacity)?;
            cuda_malloc(&mut self.compiled.eval_counts, capacity)?;
            cuda_malloc(&mut self.compiled.connection_counts, capacity)?;
            Ok(())
        })();

        if result.is_err() {
            let _ = self.clean_buffers();
        }

        result
    }

    fn copy_from(&mut self, src: &Self, num_inputs: u32) -> Result<()> {
        if self.genome.connection_stride != src.genome.connection_stride
            || self.genome.node_stride != src.genome.node_stride
            || self.compiled.node_stride != src.compiled.node_stride
            || self.compiled.connection_stride != src.compiled.connection_stride
            || self.compiled.output_stride != src.compiled.output_stride
        {
            bail!("population buffer layout mismatch during device copy")
        }

        if src.capacity > self.capacity {
            bail!("population copy source capacity {} exceeds destination capacity {}", src.capacity, self.capacity)
        }

        let copy_cap = src.capacity as usize;
        let num_inputs = num_inputs as usize;
        let connection_stride = src.genome.connection_stride as usize;
        let node_stride = src.genome.node_stride as usize;
        let output_stride = src.compiled.output_stride as usize;
        let offset_stride = node_stride.checked_add(1).context("population compiled offset stride overflowed")?;
        let sensor_entries = checked_usize_product(copy_cap, num_inputs, "population sensor copy size overflowed")?;
        let connection_entries =
            checked_usize_product(copy_cap, connection_stride, "population connection copy size overflowed")?;
        let node_entries = checked_usize_product(copy_cap, node_stride, "population node copy size overflowed")?;
        let offset_entries =
            checked_usize_product(copy_cap, offset_stride, "population compiled offset copy size overflowed")?;
        let output_entries = checked_usize_product(copy_cap, output_stride, "population output copy size overflowed")?;

        cuda_dev_to_dev(self.pos_x, src.pos_x, copy_cap)?;
        cuda_dev_to_dev(self.pos_y, src.pos_y, copy_cap)?;
        cuda_dev_to_dev(self.vel_x, src.vel_x, copy_cap)?;
        cuda_dev_to_dev(self.vel_y, src.vel_y, copy_cap)?;
        cuda_dev_to_dev(self.energy, src.energy, copy_cap)?;
        cuda_dev_to_dev(self.age, src.age, copy_cap)?;
        cuda_dev_to_dev(self.alive, src.alive, copy_cap)?;
        cuda_dev_to_dev(self.species_id, src.species_id, copy_cap)?;
        cuda_dev_to_dev(self.entity_id, src.entity_id, copy_cap)?;
        cuda_dev_to_dev(self.generation, src.generation, copy_cap)?;
        cuda_dev_to_dev(self.rng_state, src.rng_state, copy_cap)?;
        cuda_dev_to_dev(self.sensor_inputs, src.sensor_inputs, sensor_entries)?;
        cuda_dev_to_dev(self.genome.connection_from, src.genome.connection_from, connection_entries)?;
        cuda_dev_to_dev(self.genome.connection_to, src.genome.connection_to, connection_entries)?;
        cuda_dev_to_dev(self.genome.connection_weight, src.genome.connection_weight, connection_entries)?;
        cuda_dev_to_dev(self.genome.connection_innovation, src.genome.connection_innovation, connection_entries)?;
        cuda_dev_to_dev(self.genome.connection_enabled, src.genome.connection_enabled, connection_entries)?;
        cuda_dev_to_dev(self.genome.node_types, src.genome.node_types, node_entries)?;
        cuda_dev_to_dev(self.genome.num_connections, src.genome.num_connections, copy_cap)?;
        cuda_dev_to_dev(self.genome.num_nodes, src.genome.num_nodes, copy_cap)?;
        cuda_dev_to_dev(self.compiled.eval_order, src.compiled.eval_order, node_entries)?;
        cuda_dev_to_dev(self.compiled.connection_offsets, src.compiled.connection_offsets, offset_entries)?;
        cuda_dev_to_dev(self.compiled.output_indices, src.compiled.output_indices, output_entries)?;
        cuda_dev_to_dev(self.compiled.connection_sources, src.compiled.connection_sources, connection_entries)?;
        cuda_dev_to_dev(self.compiled.connection_weights, src.compiled.connection_weights, connection_entries)?;
        cuda_dev_to_dev(self.compiled.node_counts, src.compiled.node_counts, copy_cap)?;
        cuda_dev_to_dev(self.compiled.eval_counts, src.compiled.eval_counts, copy_cap)?;
        cuda_dev_to_dev(self.compiled.connection_counts, src.compiled.connection_counts, copy_cap)?;

        Ok(())
    }

    fn zero_tail(&mut self, from_capacity: u32, num_inputs: u32) -> Result<()> {
        if from_capacity >= self.capacity {
            return Ok(());
        }

        let tail = (self.capacity - from_capacity) as usize;
        let from_capacity = from_capacity as usize;
        let num_inputs = num_inputs as usize;
        let connection_stride = self.genome.connection_stride as usize;
        let node_stride = self.genome.node_stride as usize;
        let output_stride = self.compiled.output_stride as usize;
        let offset_stride = node_stride.checked_add(1).context("population compiled offset stride overflowed")?;
        let sensor_base = checked_usize_product(from_capacity, num_inputs, "population sensor tail offset overflowed")?;
        let sensor_entries = checked_usize_product(tail, num_inputs, "population sensor tail size overflowed")?;
        let connection_base =
            checked_usize_product(from_capacity, connection_stride, "population connection tail offset overflowed")?;
        let connection_entries =
            checked_usize_product(tail, connection_stride, "population connection tail size overflowed")?;
        let node_base = checked_usize_product(from_capacity, node_stride, "population node tail offset overflowed")?;
        let node_entries = checked_usize_product(tail, node_stride, "population node tail size overflowed")?;
        let offset_base =
            checked_usize_product(from_capacity, offset_stride, "population compiled offset tail offset overflowed")?;
        let offset_entries =
            checked_usize_product(tail, offset_stride, "population compiled offset tail size overflowed")?;
        let output_base =
            checked_usize_product(from_capacity, output_stride, "population output tail offset overflowed")?;
        let output_entries = checked_usize_product(tail, output_stride, "population output tail size overflowed")?;

        cuda_memset_zero(self.pos_x.wrapping_add(from_capacity), tail)?;
        cuda_memset_zero(self.pos_y.wrapping_add(from_capacity), tail)?;
        cuda_memset_zero(self.vel_x.wrapping_add(from_capacity), tail)?;
        cuda_memset_zero(self.vel_y.wrapping_add(from_capacity), tail)?;
        cuda_memset_zero(self.energy.wrapping_add(from_capacity), tail)?;
        cuda_memset_zero(self.age.wrapping_add(from_capacity), tail)?;
        cuda_memset_zero(self.alive.wrapping_add(from_capacity), tail)?;
        cuda_memset_zero(self.species_id.wrapping_add(from_capacity), tail)?;
        cuda_memset_zero(self.entity_id.wrapping_add(from_capacity), tail)?;
        cuda_memset_zero(self.generation.wrapping_add(from_capacity), tail)?;
        cuda_memset_zero(self.rng_state.wrapping_add(from_capacity), tail)?;
        cuda_memset_zero(self.sensor_inputs.wrapping_add(sensor_base), sensor_entries)?;
        cuda_memset_zero(self.genome.connection_from.wrapping_add(connection_base), connection_entries)?;
        cuda_memset_zero(self.genome.connection_to.wrapping_add(connection_base), connection_entries)?;
        cuda_memset_zero(self.genome.connection_weight.wrapping_add(connection_base), connection_entries)?;
        cuda_memset_zero(self.genome.connection_innovation.wrapping_add(connection_base), connection_entries)?;
        cuda_memset_zero(self.genome.connection_enabled.wrapping_add(connection_base), connection_entries)?;
        cuda_memset_zero(self.genome.node_types.wrapping_add(node_base), node_entries)?;
        cuda_memset_zero(self.genome.num_connections.wrapping_add(from_capacity), tail)?;
        cuda_memset_zero(self.genome.num_nodes.wrapping_add(from_capacity), tail)?;
        cuda_memset_zero(self.compiled.eval_order.wrapping_add(node_base), node_entries)?;
        cuda_memset_zero(self.compiled.connection_offsets.wrapping_add(offset_base), offset_entries)?;
        cuda_memset_zero(self.compiled.output_indices.wrapping_add(output_base), output_entries)?;
        cuda_memset_zero(self.compiled.connection_sources.wrapping_add(connection_base), connection_entries)?;
        cuda_memset_zero(self.compiled.connection_weights.wrapping_add(connection_base), connection_entries)?;
        cuda_memset_zero(self.compiled.node_counts.wrapping_add(from_capacity), tail)?;
        cuda_memset_zero(self.compiled.eval_counts.wrapping_add(from_capacity), tail)?;
        cuda_memset_zero(self.compiled.connection_counts.wrapping_add(from_capacity), tail)?;

        Ok(())
    }

    fn clean_buffers(&mut self) -> Result<()> {
        cuda_free(&mut self.pos_x)?;
        cuda_free(&mut self.pos_y)?;
        cuda_free(&mut self.vel_x)?;
        cuda_free(&mut self.vel_y)?;
        cuda_free(&mut self.energy)?;
        cuda_free(&mut self.age)?;
        cuda_free(&mut self.alive)?;
        cuda_free(&mut self.species_id)?;
        cuda_free(&mut self.entity_id)?;
        cuda_free(&mut self.generation)?;
        cuda_free(&mut self.rng_state)?;
        cuda_free(&mut self.sensor_inputs)?;
        cuda_free(&mut self.genome.connection_from)?;
        cuda_free(&mut self.genome.connection_to)?;
        cuda_free(&mut self.genome.connection_weight)?;
        cuda_free(&mut self.genome.connection_innovation)?;
        cuda_free(&mut self.genome.connection_enabled)?;
        cuda_free(&mut self.genome.node_types)?;
        cuda_free(&mut self.genome.num_connections)?;
        cuda_free(&mut self.genome.num_nodes)?;
        cuda_free(&mut self.compiled.eval_order)?;
        cuda_free(&mut self.compiled.connection_offsets)?;
        cuda_free(&mut self.compiled.output_indices)?;
        cuda_free(&mut self.compiled.connection_sources)?;
        cuda_free(&mut self.compiled.connection_weights)?;
        cuda_free(&mut self.compiled.node_counts)?;
        cuda_free(&mut self.compiled.eval_counts)?;
        cuda_free(&mut self.compiled.connection_counts)?;
        self.capacity = 0;

        Ok(())
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeviceInnovationState {
    next_innovation: u32,
    next_node_id: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimulationCounters {
    tick: u32,
    predator_births: u32,
    prey_births: u32,
    predator_deaths: u32,
    prey_deaths: u32,
    kills: u32,
    food_eaten: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PopulationGridEntry {
    slot: u32,
    pos_x: f32,
    pos_y: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FoodGridEntry {
    slot: u32,
    pos_x: f32,
    pos_y: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricsReduceScratch {
    predator_energy_sum: f32,
    prey_energy_sum: f32,
    predator_complexity_sum: f32,
    prey_complexity_sum: f32,
    predator_generation_sum: f32,
    prey_generation_sum: f32,
    predator_count: u32,
    prey_count: u32,
    max_predator_generation: u32,
    max_prey_generation: u32,
    predator_species_mask: u64,
    prey_species_mask: u64,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct UiStatsReduceScratch {
    predator_energy_sum: f32,
    prey_energy_sum: f32,
    predator_count: u32,
    prey_count: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct FoodBuffer {
    pos_x: *mut f32,
    pos_y: *mut f32,
    active: *mut u8,
    capacity: u32,
}

impl FoodBuffer {
    fn allocate(&mut self, capacity: u32) -> Result<()> {
        self.capacity = capacity;

        let result = (|| -> Result<()> {
            let capacity = capacity as usize;
            cuda_malloc(&mut self.pos_x, capacity)?;
            cuda_malloc(&mut self.pos_y, capacity)?;
            cuda_malloc(&mut self.active, capacity)?;
            Ok(())
        })();

        if result.is_err() {
            let _ = self.clean_buffers();
        }

        result
    }

    fn clean_buffers(&mut self) -> Result<()> {
        cuda_free(&mut self.pos_x)?;
        cuda_free(&mut self.pos_y)?;
        cuda_free(&mut self.active)?;
        self.capacity = 0;

        Ok(())
    }
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct DeviceState {
    simulation: SimulationConfig,
    predator: DevicePopulationBuffers,
    prey: DevicePopulationBuffers,
    food: FoodBuffer,
    innovation: *mut DeviceInnovationState,
    next_entity_id: *mut u32,
    counters: *mut SimulationCounters,
    predator_free_list: *mut u32,
    prey_free_list: *mut u32,
    predator_free_len: *mut u32,
    prey_free_len: *mut u32,
    predator_mate_claims: *mut u32,
    prey_mate_claims: *mut u32,
    predator_reproduction_pairs: *mut ReproductionPairReadback,
    prey_reproduction_pairs: *mut ReproductionPairReadback,
    predator_pair_count: *mut u32,
    prey_pair_count: *mut u32,
    population_live_count_scratch: *mut u32,
    ui_stats_scratch: *mut UiStatsReadback,
    ui_stats_reduce_scratch: *mut UiStatsReduceScratch,
    free_list_state_scratch: *mut FreeListStateReadback,
    sensor_snapshot_scratch: *mut SensorSnapshotReadback,
    compiled_header_scratch: *mut CompiledNetworkReadbackHeader,
    selected_network_scratch: *mut SelectedAgentNetworkReadback,
    metrics_summary: *mut MetricsSummaryReadback,
    metrics_reduce_scratch: *mut MetricsReduceScratch,
    species_summaries_scratch: *mut SpeciesSummaryReadback,
    representative_headers_scratch: *mut RepresentativeGenomeHeader,
    species_count_scratch: *mut u32,
    render_header_scratch: *mut RenderSnapshotHeader,
    render_predators_scratch: *mut RenderAgentReadback,
    render_prey_scratch: *mut RenderAgentReadback,
    render_food_scratch: *mut RenderFoodReadback,
    predator_cell_counts: *mut u32,
    predator_cell_offsets: *mut u32,
    predator_cell_write_offsets: *mut u32,
    predator_grid_entries: *mut PopulationGridEntry,
    prey_cell_counts: *mut u32,
    prey_cell_offsets: *mut u32,
    prey_cell_write_offsets: *mut u32,
    prey_grid_entries: *mut PopulationGridEntry,
    food_cell_counts: *mut u32,
    food_cell_offsets: *mut u32,
    food_cell_write_offsets: *mut u32,
    food_grid_entries: *mut FoodGridEntry,
    food_claimed_by: *mut u32,
    prey_claimed_by: *mut u32,
    num_inputs: u32,
    num_outputs: u32,
    node_stride: u32,
    connection_stride: u32,
    grid_cols: u32,
    grid_rows: u32,
    grid_cell_capacity: u32,
    grid_cell_size: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpeciesSummaryReadback {
    pub population_kind: PopulationKind,
    pub species_id: u32,
    pub size: u32,
    pub representative_slot: u32,
    pub avg_complexity: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepresentativeGenomeHeader {
    pub population_kind: PopulationKind,
    pub slot: u32,
    pub entity_id: u32,
    pub generation: u32,
    pub species_id: u32,
    pub num_nodes: u16,
    pub num_connections: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenomeNodeReadback {
    pub id: u32,
    pub node_type: u8,
    pub reserved0: u8,
    pub reserved1: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GenomeConnectionReadback {
    pub from_node: i32,
    pub to_node: i32,
    pub weight: f32,
    pub innovation: u32,
    pub enabled: u8,
    pub reserved0: u8,
    pub reserved1: u16,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepresentativeGenomeReadback {
    pub header: RepresentativeGenomeHeader,
    pub nodes: Vec<GenomeNodeReadback>,
    pub connections: Vec<GenomeConnectionReadback>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeciesBatchReadbackHeader {
    pub population_kind: PopulationKind,
    pub species_count: u32,
    pub returned_species_count: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FreeListStateReadback {
    pub tick: u32,
    pub predator_free_slots: u32,
    pub prey_free_slots: u32,
    pub active_food_count: u32,
    pub food_capacity: u32,
}

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

#[derive(Debug, Clone, PartialEq)]
pub struct RenderSnapshotReadback {
    pub header: RenderSnapshotHeader,
    pub predators: Vec<RenderAgentReadback>,
    pub prey: Vec<RenderAgentReadback>,
    pub food: Vec<RenderFoodReadback>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledNetworkReadbackHeader {
    pub population_kind: PopulationKind,
    pub slot: u32,
    pub node_count: u16,
    pub eval_node_count: u16,
    pub output_count: u16,
    pub connection_count: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SensorSnapshotReadback {
    pub population_kind: PopulationKind,
    pub slot: u32,
    pub input_count: u16,
    pub reserved: u16,
    pub inputs: [f32; 35],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SelectedAgentNetworkReadback {
    pub population_kind: PopulationKind,
    pub slot: u32,
    pub node_count: u16,
    pub output_count: u16,
    pub activation_count: u16,
    pub reserved: u16,
    pub output_0: f32,
    pub output_1: f32,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PopulationKind {
    Predator = 0,
    Prey = 1,
}

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

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReproductionPairReadback {
    pub parent_a_slot: u32,
    pub parent_b_slot: u32,
}

fn device_read<T>(context: &str, device_ptr: *const T) -> Result<T> {
    let mut out = MaybeUninit::<T>::uninit();
    cuda_dev_to_host(device_ptr, out.as_mut_ptr(), 1).with_context(|| context.to_owned())?;
    Ok(unsafe { out.assume_init() })
}

fn device_read_slice<T>(context: &str, device_ptr: *const T, out: &mut [T]) -> Result<()> {
    cuda_dev_to_host(device_ptr, out.as_mut_ptr(), out.len()).with_context(|| context.to_owned())
}

pub struct Simulation {
    pub config: SimulationConfig,
    device_state: DeviceState,
}

impl Drop for Simulation {
    fn drop(&mut self) {
        if let Ok(()) = self.clean_buffers() {}
    }
}

impl Simulation {
    pub fn init(config: &SimulationConfig) -> Result<Self> {
        let mut simulation = Simulation { device_state: DeviceState::default(), config: *config };
        simulation.init_dev()?;
        Ok(simulation)
    }

    pub fn tick(&mut self) -> Result<()> {
        profile_scope!("tick");

        self.build_spatial_grid()?;
        self.compute_sensor_inputs()?;
        self.infer_populations()?;
        self.update_population_vitals()?;
        self.apply_population_movement()?;
        self.build_spatial_grid()?;
        self.resolve_food()?;
        self.resolve_combat()?;
        let mut predator_births = self.reproduction_candidate_count(PopulationKind::Predator)?;
        if self.ensure_birth_capacity(PopulationKind::Predator, predator_births)? {
            predator_births = self.reproduction_candidate_count(PopulationKind::Predator)?;
        }
        self.run_reproduction(PopulationKind::Predator, predator_births)?;
        let mut prey_births = self.reproduction_candidate_count(PopulationKind::Prey)?;
        if self.ensure_birth_capacity(PopulationKind::Prey, prey_births)? {
            prey_births = self.reproduction_candidate_count(PopulationKind::Prey)?;
        }
        self.run_reproduction(PopulationKind::Prey, prey_births)?;
        self.advance_tick()?;

        Ok(())
    }

    pub fn ui_stats(&mut self) -> Result<UiStatsReadback> {
        self.read_ui_stats()
    }

    pub fn free_list_state(&mut self) -> Result<FreeListStateReadback> {
        self.read_free_list_state()
    }

    pub fn metrics_summary(&mut self) -> Result<MetricsSummaryReadback> {
        self.read_metrics_summary()
    }

    pub fn refresh_reports(&mut self) -> Result<()> {
        self.refresh_reports_impl()
    }

    pub fn sensor_snapshot(&mut self, population_kind: PopulationKind, slot: u32) -> Result<SensorSnapshotReadback> {
        self.read_sensor_snapshot(population_kind, slot)
    }

    pub fn render_snapshot(
        &mut self,
        max_predators: u32,
        max_prey: u32,
        max_food: u32,
    ) -> Result<RenderSnapshotReadback> {
        self.read_render_snapshot(max_predators, max_prey, max_food)
    }

    pub fn selected_agent_network(
        &mut self,
        population_kind: PopulationKind,
        slot: u32,
    ) -> Result<SelectedAgentNetworkReadback> {
        self.read_selected_agent_network(population_kind, slot)
    }

    pub fn species_summaries(
        &mut self,
        population_kind: PopulationKind,
        max_species: u32,
    ) -> Result<(SpeciesBatchReadbackHeader, Vec<SpeciesSummaryReadback>, Vec<RepresentativeGenomeHeader>)> {
        self.collect_species_summaries(population_kind, max_species)
    }

    pub fn representative_genome(
        &mut self,
        population_kind: PopulationKind,
        slot: u32,
    ) -> Result<RepresentativeGenomeReadback> {
        self.representative_genome_impl(population_kind, slot)
    }

    const fn population_capacity(&self, population_kind: PopulationKind) -> u32 {
        match population_kind {
            PopulationKind::Predator => self.device_state.predator.capacity,
            PopulationKind::Prey => self.device_state.prey.capacity,
        }
    }

    const fn mutation_config(&self) -> Result<GpuMutationConfig> {
        Ok(GpuMutationConfig {
            mutation_rate: self.config.mutation_rate,
            weight_mutation_power: self.config.weight_mutation_power,
            add_node_rate: self.config.add_node_rate,
            add_connection_rate: self.config.add_connection_rate,
            delete_connection_rate: self.config.delete_connection_rate,
            max_connection_attempts: 16,
        })
    }

    const unsafe fn get_dev_state(&mut self) -> *mut DeviceState {
        &mut self.device_state as *mut DeviceState
    }
}
