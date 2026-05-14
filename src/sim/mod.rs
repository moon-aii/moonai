mod helpers;
use crate::sim::helpers::*;

use crate::experiment::SimulationConfig;
use crate::profile_scope;
use anyhow::{Context as _, Result, bail};
use serde::{Deserialize, Serialize};
use std::mem::MaybeUninit;
use std::ptr::{self};

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

    fn allocate_core_state_buffers(&mut self) -> Result<()> {
        self.device_state.predator.allocate(
            self.config.predator_count,
            self.device_state.num_inputs,
            self.device_state.node_stride,
            self.device_state.connection_stride,
            self.device_state.num_outputs,
        )?;
        self.device_state.prey.allocate(
            self.config.prey_count,
            self.device_state.num_inputs,
            self.device_state.node_stride,
            self.device_state.connection_stride,
            self.device_state.num_outputs,
        )?;
        cuda_malloc(&mut self.device_state.innovation, 1)?;
        cuda_malloc(&mut self.device_state.next_entity_id, 1)?;
        cuda_malloc(&mut self.device_state.population_live_count_scratch, 1)?;
        cuda_malloc(&mut self.device_state.ui_stats_scratch, 1)?;
        cuda_malloc(&mut self.device_state.free_list_state_scratch, 1)?;
        cuda_malloc(&mut self.device_state.sensor_snapshot_scratch, 1)?;
        cuda_malloc(&mut self.device_state.compiled_header_scratch, 1)?;
        cuda_malloc(&mut self.device_state.selected_network_scratch, 1)?;
        cuda_malloc(&mut self.device_state.metrics_summary, 1)?;
        cuda_malloc(&mut self.device_state.metrics_reduce_scratch, 1)?;
        cuda_malloc(&mut self.device_state.species_summaries_scratch, MAX_SPECIES_SUMMARIES as usize)?;
        cuda_malloc(&mut self.device_state.representative_headers_scratch, MAX_SPECIES_SUMMARIES as usize)?;
        cuda_malloc(&mut self.device_state.species_count_scratch, 1)?;
        cuda_malloc(&mut self.device_state.render_header_scratch, 1)?;

        Ok(())
    }

    fn allocate_population_render_buffers(&mut self) -> Result<()> {
        let result = (|| -> Result<()> {
            cuda_malloc(&mut self.device_state.render_predators_scratch, self.device_state.predator.capacity as usize)?;
            cuda_malloc(&mut self.device_state.render_prey_scratch, self.device_state.prey.capacity as usize)?;
            Ok(())
        })();

        if result.is_err() {
            let _ = cuda_free(&mut self.device_state.render_predators_scratch);
            let _ = cuda_free(&mut self.device_state.render_prey_scratch);
        }

        result
    }

    fn allocate_food_state_buffers(&mut self) -> Result<()> {
        let result = (|| -> Result<()> {
            self.device_state.food.allocate(self.config.food_count)?;
            cuda_malloc(&mut self.device_state.render_food_scratch, self.device_state.food.capacity as usize)?;
            cuda_malloc(&mut self.device_state.food_claimed_by, self.device_state.food.capacity as usize)?;
            Ok(())
        })();

        if result.is_err() {
            let _ = self.device_state.food.clean_buffers();
            let _ = cuda_free(&mut self.device_state.render_food_scratch);
            let _ = cuda_free(&mut self.device_state.food_claimed_by);
        }

        result
    }

    fn allocate_counter_buffer(&mut self) -> Result<()> {
        cuda_malloc(&mut self.device_state.counters, 1)
    }

    fn free_free_list_buffers(&mut self) -> Result<()> {
        cuda_free(&mut self.device_state.predator_free_list)?;
        cuda_free(&mut self.device_state.prey_free_list)?;
        cuda_free(&mut self.device_state.predator_free_len)?;
        cuda_free(&mut self.device_state.prey_free_len)?;
        cuda_free(&mut self.device_state.prey_claimed_by)?;

        Ok(())
    }

    fn allocate_free_list_buffers(&mut self) -> Result<()> {
        let result = (|| -> Result<()> {
            cuda_malloc(&mut self.device_state.predator_free_list, self.device_state.predator.capacity as usize)?;
            cuda_malloc(&mut self.device_state.prey_free_list, self.device_state.prey.capacity as usize)?;
            cuda_malloc(&mut self.device_state.predator_free_len, 1)?;
            cuda_malloc(&mut self.device_state.prey_free_len, 1)?;
            cuda_malloc(&mut self.device_state.prey_claimed_by, self.device_state.prey.capacity as usize)?;
            Ok(())
        })();

        if result.is_err() {
            let _ = self.free_free_list_buffers();
        }

        result
    }

    fn allocate_reproduction_buffers(&mut self) -> Result<()> {
        let result = (|| -> Result<()> {
            cuda_malloc(&mut self.device_state.predator_mate_claims, self.device_state.predator.capacity as usize)?;
            cuda_malloc(&mut self.device_state.prey_mate_claims, self.device_state.prey.capacity as usize)?;
            cuda_malloc(
                &mut self.device_state.predator_reproduction_pairs,
                self.device_state.predator.capacity as usize,
            )?;
            cuda_malloc(&mut self.device_state.prey_reproduction_pairs, self.device_state.prey.capacity as usize)?;
            cuda_malloc(&mut self.device_state.predator_pair_count, 1)?;
            cuda_malloc(&mut self.device_state.prey_pair_count, 1)?;
            Ok(())
        })();

        if result.is_err() {
            let _ = self.free_reproduction_buffers();
        }

        result
    }

    fn allocate_spatial_grid_buffers(&mut self) -> Result<()> {
        let grid_cols = self.device_state.grid_cols;
        let grid_rows = self.device_state.grid_rows;
        let grid_cell_size = self.device_state.grid_cell_size;
        let cell_count = grid_cols.checked_mul(grid_rows).context("spatial grid cell count overflowed")?;
        let offset_count = cell_count.checked_add(1).context("spatial grid offset count overflowed")?;

        let result = (|| -> Result<()> {
            cuda_malloc(&mut self.device_state.predator_cell_counts, cell_count as usize)?;
            cuda_malloc(&mut self.device_state.predator_cell_offsets, offset_count as usize)?;
            cuda_malloc(&mut self.device_state.predator_cell_write_offsets, cell_count as usize)?;
            cuda_malloc(&mut self.device_state.predator_grid_entries, self.device_state.predator.capacity as usize)?;
            cuda_malloc(&mut self.device_state.prey_cell_counts, cell_count as usize)?;
            cuda_malloc(&mut self.device_state.prey_cell_offsets, offset_count as usize)?;
            cuda_malloc(&mut self.device_state.prey_cell_write_offsets, cell_count as usize)?;
            cuda_malloc(&mut self.device_state.prey_grid_entries, self.device_state.prey.capacity as usize)?;
            cuda_malloc(&mut self.device_state.food_cell_counts, cell_count as usize)?;
            cuda_malloc(&mut self.device_state.food_cell_offsets, offset_count as usize)?;
            cuda_malloc(&mut self.device_state.food_cell_write_offsets, cell_count as usize)?;
            cuda_malloc(&mut self.device_state.food_grid_entries, self.device_state.food.capacity as usize)?;
            Ok(())
        })();

        if result.is_err() {
            let _ = self.clean_spatial_grid_buffers();
            self.device_state.grid_cols = grid_cols;
            self.device_state.grid_rows = grid_rows;
            self.device_state.grid_cell_size = grid_cell_size;
        } else {
            self.device_state.grid_cell_capacity = cell_count;
        }

        result
    }

    fn initialize_device_scalars(&mut self) -> Result<()> {
        let next_innovation = self
            .device_state
            .num_inputs
            .checked_add(1)
            .and_then(|value| value.checked_mul(self.device_state.num_outputs))
            .context("innovation counter initialization overflowed")?;
        let next_node_id = self
            .device_state
            .num_inputs
            .checked_add(self.device_state.num_outputs)
            .and_then(|value| value.checked_add(1))
            .context("node id initialization overflowed")?;
        let innovation_state = DeviceInnovationState { next_innovation, next_node_id };
        let next_entity_id = self
            .config
            .predator_count
            .checked_add(self.config.prey_count)
            .and_then(|value| value.checked_add(1))
            .context("next entity id initialization overflowed")?;

        cuda_host_to_dev(self.device_state.innovation, ptr::from_ref(&innovation_state), 1)?;
        cuda_host_to_dev(self.device_state.next_entity_id, ptr::from_ref(&next_entity_id), 1)
    }

    fn init_dev(&mut self) -> Result<()> {
        self.device_state.simulation = self.config;
        self.device_state.grid_cell_size = self.config.vision_range.max(1.0);
        self.device_state.grid_cols = ((self.config.grid_size / self.device_state.grid_cell_size).ceil() as u32).max(1);
        self.device_state.grid_rows = ((self.config.grid_size / self.device_state.grid_cell_size).ceil() as u32).max(1);

        let hidden_budget = self.config.max_hidden_nodes;

        let seeded_node_count = SENSOR_COUNT
            .checked_add(OUTPUT_COUNT)
            .and_then(|value| value.checked_add(1))
            .context("seed-stage node stride overflowed")?;
        let seeded_connection_count = SENSOR_COUNT
            .checked_add(1)
            .and_then(|value| value.checked_mul(OUTPUT_COUNT))
            .context("seed-stage connection stride overflowed")?;
        let extra_connection_capacity = hidden_budget.saturating_mul(2).clamp(4, PHASE3_CONNECTION_GROWTH_BUDGET_CAP);

        let node_stride = seeded_node_count.checked_add(hidden_budget).context("phase-3 node stride overflowed")?;
        let connection_stride = seeded_connection_count
            .checked_add(extra_connection_capacity)
            .context("phase-3 connection stride overflowed")?;

        self.device_state.num_inputs = SENSOR_COUNT;
        self.device_state.num_outputs = OUTPUT_COUNT;
        self.device_state.node_stride = node_stride;
        self.device_state.connection_stride = connection_stride;

        self.allocate_core_state_buffers()?;
        self.allocate_population_render_buffers()?;
        self.allocate_food_state_buffers()?;
        self.allocate_counter_buffer()?;
        self.allocate_free_list_buffers()?;
        self.allocate_reproduction_buffers()?;
        self.allocate_spatial_grid_buffers()?;
        check_cuda_status(unsafe { dev_seed_initial_population(self.get_dev_state()) }, "dev_seed_initial_population")?;
        self.initialize_device_scalars()?;
        check_cuda_status(unsafe { dev_reset_counters(self.get_dev_state()) }, "dev_reset_counters")?;
        check_cuda_status(
            unsafe { dev_initialize_population_free_list(self.get_dev_state(), PopulationKind::Predator) },
            "dev_initialize_predator_free_list",
        )?;
        check_cuda_status(
            unsafe { dev_initialize_population_free_list(self.get_dev_state(), PopulationKind::Prey) },
            "dev_initialize_prey_free_list",
        )?;
        check_cuda_status(unsafe { dev_seed_food(self.get_dev_state()) }, "dev_seed_food")?;
        check_cuda_status(
            unsafe { dev_reset_population_reproduction_state(self.get_dev_state(), PopulationKind::Predator) },
            "dev_reset_predator_reproduction_state",
        )?;
        check_cuda_status(
            unsafe { dev_reset_population_reproduction_state(self.get_dev_state(), PopulationKind::Prey) },
            "dev_reset_prey_reproduction_state",
        )?;

        if self.device_state.predator.capacity > 0 {
            let _ = self.compile_population(PopulationKind::Predator, 0)?;
        }
        if self.device_state.prey.capacity > 0 {
            let _ = self.compile_population(PopulationKind::Prey, 0)?;
        }

        self.refresh_reports()?;

        Ok(())
    }

    pub fn tick(&mut self) -> Result<UiStatsReadback> {
        profile_scope!("tick");

        self.build_spatial_grid()?;
        self.compute_sensor_inputs()?;
        self.infer_population(PopulationKind::Predator)?;
        self.infer_population(PopulationKind::Prey)?;
        self.update_vitals(PopulationKind::Predator)?;
        self.update_vitals(PopulationKind::Prey)?;
        self.apply_movement(PopulationKind::Predator)?;
        self.apply_movement(PopulationKind::Prey)?;
        self.build_spatial_grid()?;
        self.resolve_food()?;
        self.resolve_combat()?;
        let predator_births = self.reproduction_candidate_count(PopulationKind::Predator)?;
        self.ensure_birth_capacity(PopulationKind::Predator, predator_births)?;
        self.run_reproduction(PopulationKind::Predator)?;
        let prey_births = self.reproduction_candidate_count(PopulationKind::Prey)?;
        self.ensure_birth_capacity(PopulationKind::Prey, prey_births)?;
        self.run_reproduction(PopulationKind::Prey)?;
        self.advance_tick()?;

        let ui_stats = self.ui_stats()?;
        if self.config.report_interval_ticks > 0 && ui_stats.tick % self.config.report_interval_ticks == 0 {
            self.refresh_reports()?;
        }
        Ok(ui_stats)
    }

    pub fn ui_stats(&mut self) -> Result<UiStatsReadback> {
        profile_scope!("ui_stats");
        let status = unsafe { dev_write_ui_stats(self.get_dev_state()) };
        check_cuda_status(status, "moonai_gpu_simulation_ui_stats")?;
        device_read("moonai_gpu_simulation_ui_stats_readback", self.device_state.ui_stats_scratch)
    }

    pub fn free_list_state(&mut self) -> Result<FreeListStateReadback> {
        let status = unsafe { dev_write_free_list_state(self.get_dev_state()) };
        check_cuda_status(status, "moonai_gpu_simulation_free_list_state")?;
        device_read("moonai_gpu_simulation_free_list_state_readback", self.device_state.free_list_state_scratch)
    }

    pub fn metrics_summary(&mut self) -> Result<MetricsSummaryReadback> {
        device_read("moonai_gpu_simulation_metrics_summary", self.device_state.metrics_summary)
    }

    pub fn refresh_reports(&mut self) -> Result<()> {
        profile_scope!("refresh_reports");
        check_cuda_status(
            unsafe { dev_classify_species_summaries(self.get_dev_state(), PopulationKind::Predator) },
            "moonai_gpu_simulation_classify_predator_species",
        )?;
        check_cuda_status(
            unsafe { dev_classify_species_summaries(self.get_dev_state(), PopulationKind::Prey) },
            "moonai_gpu_simulation_classify_prey_species",
        )?;
        cuda_memset_zero(self.device_state.metrics_reduce_scratch, 1)
            .context("moonai_gpu_simulation_zero_metrics_reduce_scratch")?;
        check_cuda_status(
            unsafe { dev_accumulate_metrics(self.get_dev_state(), PopulationKind::Predator) },
            "moonai_gpu_simulation_accumulate_predator_metrics",
        )?;
        check_cuda_status(
            unsafe { dev_accumulate_metrics(self.get_dev_state(), PopulationKind::Prey) },
            "moonai_gpu_simulation_accumulate_prey_metrics",
        )?;
        let status = unsafe { dev_finalize_metrics_summary(self.get_dev_state()) };
        check_cuda_status(status, "moonai_gpu_simulation_finalize_metrics_summary")
    }

    pub fn sensor_snapshot(&mut self, population_kind: PopulationKind, slot: u32) -> Result<SensorSnapshotReadback> {
        let status = unsafe { dev_write_sensor_snapshot(self.get_dev_state(), population_kind, slot) };
        check_cuda_status(status, "moonai_gpu_simulation_sensor_snapshot")?;
        device_read("moonai_gpu_simulation_sensor_snapshot_readback", self.device_state.sensor_snapshot_scratch)
    }

    pub fn render_snapshot(
        &mut self,
        max_predators: u32,
        max_prey: u32,
        max_food: u32,
    ) -> Result<RenderSnapshotReadback> {
        let predator_capacity = usize::try_from(max_predators).context("predator render capacity overflowed")?;
        let prey_capacity = usize::try_from(max_prey).context("prey render capacity overflowed")?;
        let food_capacity = usize::try_from(max_food).context("food render capacity overflowed")?;
        let empty_predator = RenderAgentReadback {
            population_kind: PopulationKind::Predator,
            slot: 0,
            entity_id: 0,
            species_id: 0,
            generation: 0,
            age: 0.0,
            pos_x: 0.0,
            pos_y: 0.0,
            dir_x: 0.0,
            dir_y: 0.0,
            energy: 0.0,
        };
        let empty_prey = RenderAgentReadback { population_kind: PopulationKind::Prey, ..empty_predator };
        let empty_food = RenderFoodReadback { slot: 0, active: 0, reserved0: 0, reserved1: 0, pos_x: 0.0, pos_y: 0.0 };
        let mut predators = vec![empty_predator; predator_capacity];
        let mut prey = vec![empty_prey; prey_capacity];
        let mut food = vec![empty_food; food_capacity];
        check_cuda_status(
            unsafe { dev_initialize_render_snapshot(self.get_dev_state()) },
            "moonai_gpu_simulation_initialize_render_snapshot",
        )?;
        check_cuda_status(
            unsafe { dev_pack_render_agents(self.get_dev_state(), PopulationKind::Predator, max_predators) },
            "moonai_gpu_simulation_pack_predator_render_snapshot",
        )?;
        check_cuda_status(
            unsafe { dev_pack_render_agents(self.get_dev_state(), PopulationKind::Prey, max_prey) },
            "moonai_gpu_simulation_pack_prey_render_snapshot",
        )?;
        check_cuda_status(
            unsafe { dev_pack_render_food(self.get_dev_state(), max_food) },
            "moonai_gpu_simulation_pack_food_render_snapshot",
        )?;
        let mut header =
            device_read("moonai_gpu_simulation_render_snapshot_header", self.device_state.render_header_scratch)?;
        if header.returned_predators > max_predators {
            header.returned_predators = max_predators;
        }
        if header.returned_prey > max_prey {
            header.returned_prey = max_prey;
        }
        if header.returned_food > max_food {
            header.returned_food = max_food;
        }
        let returned_predators =
            usize::try_from(header.returned_predators).context("predator render length overflowed")?;
        let returned_prey = usize::try_from(header.returned_prey).context("prey render length overflowed")?;
        let returned_food = usize::try_from(header.returned_food).context("food render length overflowed")?;
        if returned_predators > predators.len() || returned_prey > prey.len() || returned_food > food.len() {
            bail!(
                "render snapshot returned more entries than allocated: predators {} / {}, prey {} / {}, food {} / {}",
                returned_predators,
                predators.len(),
                returned_prey,
                prey.len(),
                returned_food,
                food.len()
            );
        }
        if returned_predators > 0 {
            device_read_slice(
                "moonai_gpu_simulation_render_snapshot_predators",
                self.device_state.render_predators_scratch,
                &mut predators[..returned_predators],
            )?;
        }
        if returned_prey > 0 {
            device_read_slice(
                "moonai_gpu_simulation_render_snapshot_prey",
                self.device_state.render_prey_scratch,
                &mut prey[..returned_prey],
            )?;
        }
        if returned_food > 0 {
            device_read_slice(
                "moonai_gpu_simulation_render_snapshot_food",
                self.device_state.render_food_scratch,
                &mut food[..returned_food],
            )?;
        }
        predators.truncate(returned_predators);
        prey.truncate(returned_prey);
        food.truncate(returned_food);
        Ok(RenderSnapshotReadback { header, predators, prey, food })
    }

    pub fn selected_agent_network(
        &mut self,
        population_kind: PopulationKind,
        slot: u32,
    ) -> Result<SelectedAgentNetworkReadback> {
        let status = unsafe { dev_write_selected_agent_network(self.get_dev_state(), population_kind, slot) };
        check_cuda_status(status, "moonai_gpu_evolution_selected_agent_network")?;
        device_read("moonai_gpu_evolution_selected_agent_network_readback", self.device_state.selected_network_scratch)
    }

    pub fn species_summaries(
        &mut self,
        population_kind: PopulationKind,
        max_species: u32,
    ) -> Result<(SpeciesBatchReadbackHeader, Vec<SpeciesSummaryReadback>, Vec<RepresentativeGenomeHeader>)> {
        let species_capacity = usize::try_from(max_species).context("species summary capacity overflowed")?;
        let empty_summary = SpeciesSummaryReadback {
            population_kind,
            species_id: 0,
            size: 0,
            representative_slot: 0,
            avg_complexity: 0.0,
        };
        let empty_representative = RepresentativeGenomeHeader {
            population_kind,
            slot: 0,
            entity_id: 0,
            generation: 0,
            species_id: 0,
            num_nodes: 0,
            num_connections: 0,
        };
        let mut summaries = vec![empty_summary; species_capacity];
        let mut representatives = vec![empty_representative; species_capacity];
        let status = unsafe { dev_classify_species_summaries(self.get_dev_state(), population_kind) };
        check_cuda_status(status, "moonai_gpu_evolution_species_summaries")?;
        let species_count =
            device_read::<u32>("moonai_gpu_evolution_species_summary_count", self.device_state.species_count_scratch)?;
        let returned_species_count = species_count.min(max_species);
        let returned_len =
            usize::try_from(returned_species_count).context("species summary returned length overflowed")?;
        if returned_len > summaries.len() || returned_len > representatives.len() {
            bail!(
                "species summary returned {} entries but the host buffers only allocated {} summary slots and {} representative slots",
                returned_len,
                summaries.len(),
                representatives.len()
            );
        }
        if returned_len > 0 {
            device_read_slice(
                "moonai_gpu_evolution_species_summaries_readback",
                self.device_state.species_summaries_scratch,
                &mut summaries[..returned_len],
            )?;
            device_read_slice(
                "moonai_gpu_evolution_species_representatives_readback",
                self.device_state.representative_headers_scratch,
                &mut representatives[..returned_len],
            )?;
        }
        summaries.truncate(returned_len);
        representatives.truncate(returned_len);
        let header = SpeciesBatchReadbackHeader { population_kind, species_count, returned_species_count };
        Ok((header, summaries, representatives))
    }

    pub fn representative_genome(
        &mut self,
        population_kind: PopulationKind,
        slot: u32,
    ) -> Result<RepresentativeGenomeReadback> {
        let population = match population_kind {
            PopulationKind::Predator => self.device_state.predator,
            PopulationKind::Prey => self.device_state.prey,
        };
        let slot_index = slot as usize;
        let node_stride = population.genome.node_stride as usize;
        let connection_stride = population.genome.connection_stride as usize;
        let entity_id = device_read(
            "moonai_gpu_evolution_representative_genome_entity_id",
            population.entity_id.wrapping_add(slot_index),
        )?;
        let generation = device_read(
            "moonai_gpu_evolution_representative_genome_generation",
            population.generation.wrapping_add(slot_index),
        )?;
        let species_id = device_read(
            "moonai_gpu_evolution_representative_genome_species_id",
            population.species_id.wrapping_add(slot_index),
        )?;
        let num_nodes = device_read(
            "moonai_gpu_evolution_representative_genome_num_nodes",
            population.genome.num_nodes.wrapping_add(slot_index),
        )?;
        let num_connections = device_read(
            "moonai_gpu_evolution_representative_genome_num_connections",
            population.genome.num_connections.wrapping_add(slot_index),
        )?;
        let header = RepresentativeGenomeHeader {
            population_kind,
            slot,
            entity_id,
            generation,
            species_id,
            num_nodes,
            num_connections,
        };
        let returned_nodes = usize::from(header.num_nodes);
        let returned_connections = usize::from(header.num_connections);

        let mut node_types = vec![0_u8; returned_nodes];
        if !node_types.is_empty() {
            let node_base =
                checked_usize_product(slot_index, node_stride, "representative genome node base overflowed")?;
            device_read_slice(
                "moonai_gpu_evolution_representative_genome_node_types",
                population.genome.node_types.wrapping_add(node_base),
                &mut node_types,
            )?;
        }

        let nodes = node_types
            .into_iter()
            .enumerate()
            .map(|(id, node_type)| GenomeNodeReadback { id: id as u32, node_type, reserved0: 0, reserved1: 0 })
            .collect();

        let mut from_nodes = vec![0_i32; returned_connections];
        let mut to_nodes = vec![0_i32; returned_connections];
        let mut weights = vec![0.0_f32; returned_connections];
        let mut innovations = vec![0_u32; returned_connections];
        let mut enabled_flags = vec![0_u8; returned_connections];
        if returned_connections > 0 {
            let connection_base = checked_usize_product(
                slot_index,
                connection_stride,
                "representative genome connection base overflowed",
            )?;
            device_read_slice(
                "moonai_gpu_evolution_representative_genome_connection_from",
                population.genome.connection_from.wrapping_add(connection_base),
                &mut from_nodes,
            )?;
            device_read_slice(
                "moonai_gpu_evolution_representative_genome_connection_to",
                population.genome.connection_to.wrapping_add(connection_base),
                &mut to_nodes,
            )?;
            device_read_slice(
                "moonai_gpu_evolution_representative_genome_connection_weight",
                population.genome.connection_weight.wrapping_add(connection_base),
                &mut weights,
            )?;
            device_read_slice(
                "moonai_gpu_evolution_representative_genome_connection_innovation",
                population.genome.connection_innovation.wrapping_add(connection_base),
                &mut innovations,
            )?;
            device_read_slice(
                "moonai_gpu_evolution_representative_genome_connection_enabled",
                population.genome.connection_enabled.wrapping_add(connection_base),
                &mut enabled_flags,
            )?;
        }

        let connections = from_nodes
            .into_iter()
            .zip(to_nodes)
            .zip(weights)
            .zip(innovations)
            .zip(enabled_flags)
            .map(|((((from_node, to_node), weight), innovation), enabled)| GenomeConnectionReadback {
                from_node,
                to_node,
                weight,
                innovation,
                enabled,
                reserved0: 0,
                reserved1: 0,
            })
            .collect();

        Ok(RepresentativeGenomeReadback { header, nodes, connections })
    }

    fn ensure_birth_capacity(&mut self, population_kind: PopulationKind, births_pending: u32) -> Result<()> {
        profile_scope!("birth_cap");

        if births_pending == 0 {
            return Ok(());
        }

        let live_count = self.population_live_count(population_kind)?;
        let free_list_state = self.free_list_state()?;
        let free_slots = match population_kind {
            PopulationKind::Predator => free_list_state.predator_free_slots,
            PopulationKind::Prey => free_list_state.prey_free_slots,
        };
        let capacity = self.population_capacity(population_kind);
        let required_live = live_count.saturating_add(births_pending);
        if free_slots >= births_pending && required_live <= ((capacity * 9) / 10) {
            return Ok(());
        }

        let mut new_capacity = if capacity == 0 { 1 } else { capacity };
        while new_capacity.saturating_sub(live_count) < births_pending || required_live > ((new_capacity * 9) / 10) {
            new_capacity = if new_capacity == 0 { 1 } else { new_capacity.saturating_mul(2) };
        }
        self.expand_population(population_kind, new_capacity)?;
        self.build_spatial_grid()?;
        Ok(())
    }

    fn build_spatial_grid(&mut self) -> Result<()> {
        profile_scope!("spatial_grid");
        let cell_count = self
            .device_state
            .grid_cols
            .checked_mul(self.device_state.grid_rows)
            .context("spatial grid cell count overflowed")?;
        let cell_count = cell_count as usize;

        cuda_memset_zero(self.device_state.predator_cell_counts, cell_count)
            .context("moonai_gpu_simulation_zero_predator_cell_counts")?;
        cuda_memset_zero(self.device_state.prey_cell_counts, cell_count)
            .context("moonai_gpu_simulation_zero_prey_cell_counts")?;
        cuda_memset_zero(self.device_state.food_cell_counts, cell_count)
            .context("moonai_gpu_simulation_zero_food_cell_counts")?;

        check_cuda_status(
            unsafe { dev_count_population_cells(self.get_dev_state(), PopulationKind::Predator) },
            "moonai_gpu_simulation_count_predator_cells",
        )?;
        check_cuda_status(
            unsafe { dev_count_population_cells(self.get_dev_state(), PopulationKind::Prey) },
            "moonai_gpu_simulation_count_prey_cells",
        )?;
        check_cuda_status(
            unsafe { dev_count_food_cells(self.get_dev_state()) },
            "moonai_gpu_simulation_count_food_cells",
        )?;

        check_cuda_status(
            unsafe {
                dev_exclusive_scan_u32(
                    self.device_state.predator_cell_counts,
                    cell_count as u32,
                    self.device_state.predator_cell_offsets,
                )
            },
            "moonai_gpu_simulation_scan_predator_cells",
        )?;
        check_cuda_status(
            unsafe {
                dev_exclusive_scan_u32(
                    self.device_state.prey_cell_counts,
                    cell_count as u32,
                    self.device_state.prey_cell_offsets,
                )
            },
            "moonai_gpu_simulation_scan_prey_cells",
        )?;
        check_cuda_status(
            unsafe {
                dev_exclusive_scan_u32(
                    self.device_state.food_cell_counts,
                    cell_count as u32,
                    self.device_state.food_cell_offsets,
                )
            },
            "moonai_gpu_simulation_scan_food_cells",
        )?;

        check_cuda_status(
            unsafe {
                dev_finalize_population_cell_offsets(self.get_dev_state(), PopulationKind::Predator, cell_count as u32)
            },
            "moonai_gpu_simulation_finalize_predator_cell_offsets",
        )?;
        check_cuda_status(
            unsafe {
                dev_finalize_population_cell_offsets(self.get_dev_state(), PopulationKind::Prey, cell_count as u32)
            },
            "moonai_gpu_simulation_finalize_prey_cell_offsets",
        )?;
        check_cuda_status(
            unsafe { dev_finalize_food_cell_offsets(self.get_dev_state(), cell_count as u32) },
            "moonai_gpu_simulation_finalize_food_cell_offsets",
        )?;

        check_cuda_status(
            unsafe { dev_scatter_population_cells(self.get_dev_state(), PopulationKind::Predator) },
            "moonai_gpu_simulation_scatter_predator_cells",
        )?;
        check_cuda_status(
            unsafe { dev_scatter_population_cells(self.get_dev_state(), PopulationKind::Prey) },
            "moonai_gpu_simulation_scatter_prey_cells",
        )?;
        let status = unsafe { dev_scatter_food_cells(self.get_dev_state()) };
        check_cuda_status(status, "moonai_gpu_simulation_scatter_food_cells")
    }

    fn compute_sensor_inputs(&mut self) -> Result<()> {
        profile_scope!("sensor_inputs");
        let status = unsafe { dev_compute_sensor_inputs(self.get_dev_state()) };
        check_cuda_status(status, "moonai_gpu_simulation_compute_sensor_inputs")
    }

    fn infer_population(&mut self, population_kind: PopulationKind) -> Result<()> {
        profile_scope!("inference");
        let status = unsafe { dev_infer_population(self.get_dev_state(), population_kind) };
        check_cuda_status(status, "moonai_gpu_simulation_infer_population")
    }

    fn update_vitals(&mut self, population_kind: PopulationKind) -> Result<()> {
        profile_scope!("update_vitals");
        let status = unsafe { dev_update_vitals(self.get_dev_state(), population_kind) };
        check_cuda_status(status, "moonai_gpu_simulation_update_vitals")
    }

    fn resolve_food(&mut self) -> Result<()> {
        profile_scope!("resolve_food");
        if self.device_state.prey.capacity == 0 || self.device_state.food.capacity == 0 {
            return Ok(());
        }

        cuda_memset_byte(self.device_state.food_claimed_by, 0xFF, self.device_state.food.capacity as usize)
            .context("moonai_gpu_simulation_reset_food_claims")?;
        check_cuda_status(
            unsafe { dev_resolve_food_claims(self.get_dev_state()) },
            "moonai_gpu_simulation_resolve_food_claims",
        )?;
        check_cuda_status(unsafe { dev_finalize_food(self.get_dev_state()) }, "moonai_gpu_simulation_finalize_food")?;
        let status = unsafe { dev_respawn_food(self.get_dev_state()) };
        check_cuda_status(status, "moonai_gpu_simulation_respawn_food")
    }

    fn resolve_combat(&mut self) -> Result<()> {
        profile_scope!("resolve_combat");
        if self.device_state.predator.capacity == 0 || self.device_state.prey.capacity == 0 {
            return Ok(());
        }

        cuda_memset_byte(self.device_state.prey_claimed_by, 0xFF, self.device_state.prey.capacity as usize)
            .context("moonai_gpu_simulation_reset_prey_claims")?;
        check_cuda_status(
            unsafe { dev_resolve_combat_claims(self.get_dev_state()) },
            "moonai_gpu_simulation_resolve_combat_claims",
        )?;
        let status = unsafe { dev_finalize_combat(self.get_dev_state()) };
        check_cuda_status(status, "moonai_gpu_simulation_finalize_combat")
    }

    fn apply_movement(&mut self, population_kind: PopulationKind) -> Result<()> {
        profile_scope!("apply_movement");
        let status = unsafe { dev_apply_movement(self.get_dev_state(), population_kind) };
        check_cuda_status(status, "moonai_gpu_simulation_apply_movement")
    }

    fn reproduction_candidate_count(&mut self, population_kind: PopulationKind) -> Result<u32> {
        profile_scope!("reprod_candidate");
        check_cuda_status(
            unsafe { dev_reset_population_reproduction_state(self.get_dev_state(), population_kind) },
            "moonai_gpu_simulation_reset_reproduction_state",
        )?;
        check_cuda_status(
            unsafe { dev_find_reproduction_pairs(self.get_dev_state(), population_kind) },
            "moonai_gpu_simulation_find_reproduction_pairs",
        )?;
        device_read(
            "moonai_gpu_simulation_reproduction_candidate_count",
            match population_kind {
                PopulationKind::Predator => self.device_state.predator_pair_count,
                PopulationKind::Prey => self.device_state.prey_pair_count,
            },
        )
    }

    fn expand_population(&mut self, population_kind: PopulationKind, new_capacity: u32) -> Result<()> {
        let source_population = match population_kind {
            PopulationKind::Predator => self.device_state.predator,
            PopulationKind::Prey => self.device_state.prey,
        };
        if new_capacity <= source_population.capacity {
            return Ok(());
        }

        let mut replacement = DevicePopulationBuffers::default();
        let replacement_result = (|| -> Result<()> {
            replacement.allocate(
                new_capacity,
                self.device_state.num_inputs,
                source_population.genome.node_stride,
                source_population.genome.connection_stride,
                source_population.compiled.output_stride,
            )?;
            replacement.zero_tail(0, self.device_state.num_inputs)?;
            replacement.copy_from(&source_population, self.device_state.num_inputs)
        })();
        if let Err(err) = replacement_result {
            let _ = replacement.clean_buffers();
            return Err(err);
        }

        match population_kind {
            PopulationKind::Predator => {
                self.device_state.predator.clean_buffers()?;
                self.device_state.predator = replacement;
            }
            PopulationKind::Prey => {
                self.device_state.prey.clean_buffers()?;
                self.device_state.prey = replacement;
            }
        }

        let grid_cols = self.device_state.grid_cols;
        let grid_rows = self.device_state.grid_rows;
        let grid_cell_size = self.device_state.grid_cell_size;

        cuda_free(&mut self.device_state.render_predators_scratch)?;
        cuda_free(&mut self.device_state.render_prey_scratch)?;
        self.allocate_population_render_buffers()?;
        self.free_free_list_buffers()?;
        self.allocate_free_list_buffers()?;
        self.free_reproduction_buffers()?;
        self.allocate_reproduction_buffers()?;
        self.clean_spatial_grid_buffers()?;
        self.device_state.grid_cols = grid_cols;
        self.device_state.grid_rows = grid_rows;
        self.device_state.grid_cell_size = grid_cell_size;
        self.allocate_spatial_grid_buffers()?;

        check_cuda_status(
            unsafe { dev_initialize_population_free_list(self.get_dev_state(), PopulationKind::Predator) },
            "moonai_gpu_simulation_initialize_predator_free_list",
        )?;
        let status = unsafe { dev_initialize_population_free_list(self.get_dev_state(), PopulationKind::Prey) };
        check_cuda_status(status, "moonai_gpu_simulation_initialize_prey_free_list")
    }

    fn run_reproduction(&mut self, population_kind: PopulationKind) -> Result<()> {
        profile_scope!("run_reprod");
        let pair_count = self.reproduction_candidate_count(population_kind)?;
        if pair_count == 0 {
            return Ok(());
        }

        let free_slots = self.reproduction_free_slots(population_kind, pair_count)?;
        if free_slots.is_empty() {
            return Ok(());
        }

        let pairs = self.reproduction_pairs(population_kind, pair_count)?;
        let mutation_config = self.mutation_config()?;
        let births_applied = pairs.len().min(free_slots.len());
        for index in 0..births_applied {
            let pair = pairs[index];
            let offspring_slot = free_slots[index];
            self.crossover_slot(population_kind, pair.parent_a_slot, pair.parent_b_slot, offspring_slot)?;
            self.mutate_slot(population_kind, offspring_slot, mutation_config)?;
            let _ = self.compile_slot(population_kind, offspring_slot)?;
        }

        self.apply_reproduction_energy(population_kind, births_applied as u32)
    }

    fn advance_tick(&mut self) -> Result<()> {
        profile_scope!("advance_tick");
        let status = unsafe { dev_advance_tick(self.get_dev_state()) };
        check_cuda_status(status, "moonai_gpu_simulation_advance_tick")
    }

    fn population_live_count(&mut self, population_kind: PopulationKind) -> Result<u32> {
        let status = unsafe { dev_write_population_live_count(self.get_dev_state(), population_kind) };
        check_cuda_status(status, "moonai_gpu_evolution_population_live_count")?;
        device_read(
            "moonai_gpu_evolution_population_live_count_readback",
            self.device_state.population_live_count_scratch,
        )
    }

    const fn population_capacity(&self, population_kind: PopulationKind) -> u32 {
        match population_kind {
            PopulationKind::Predator => self.device_state.predator.capacity,
            PopulationKind::Prey => self.device_state.prey.capacity,
        }
    }

    fn compile_population(
        &mut self,
        population_kind: PopulationKind,
        inspected_slot: u32,
    ) -> Result<CompiledNetworkReadbackHeader> {
        check_cuda_status(
            unsafe { dev_compile_population(self.get_dev_state(), population_kind) },
            "moonai_gpu_evolution_compile_population",
        )?;
        check_cuda_status(
            unsafe { dev_write_compiled_header(self.get_dev_state(), population_kind, inspected_slot) },
            "moonai_gpu_evolution_compile_population_header",
        )?;
        device_read(
            "moonai_gpu_evolution_compile_population_header_readback",
            self.device_state.compiled_header_scratch,
        )
    }

    fn compile_slot(&mut self, population_kind: PopulationKind, slot: u32) -> Result<CompiledNetworkReadbackHeader> {
        check_cuda_status(
            unsafe { dev_compile_slot(self.get_dev_state(), population_kind, slot) },
            "moonai_gpu_evolution_compile_slot",
        )?;
        check_cuda_status(
            unsafe { dev_write_compiled_header(self.get_dev_state(), population_kind, slot) },
            "moonai_gpu_evolution_compile_slot_header",
        )?;
        device_read("moonai_gpu_evolution_compile_slot_header_readback", self.device_state.compiled_header_scratch)
    }

    fn reproduction_pairs(
        &mut self,
        population_kind: PopulationKind,
        pair_count: u32,
    ) -> Result<Vec<ReproductionPairReadback>> {
        let returned_pairs = device_read::<u32>(
            "moonai_gpu_simulation_read_reproduction_pair_count",
            match population_kind {
                PopulationKind::Predator => self.device_state.predator_pair_count,
                PopulationKind::Prey => self.device_state.prey_pair_count,
            },
        )?
        .min(pair_count);
        let returned_pairs = usize::try_from(returned_pairs).context("reproduction pair length overflowed")?;
        let mut pairs = vec![ReproductionPairReadback { parent_a_slot: 0, parent_b_slot: 0 }; returned_pairs];
        if returned_pairs > 0 {
            device_read_slice(
                "moonai_gpu_simulation_read_reproduction_pairs",
                match population_kind {
                    PopulationKind::Predator => self.device_state.predator_reproduction_pairs,
                    PopulationKind::Prey => self.device_state.prey_reproduction_pairs,
                },
                &mut pairs,
            )?;
        }
        pairs.truncate(returned_pairs);
        Ok(pairs)
    }

    fn reproduction_free_slots(&mut self, population_kind: PopulationKind, slot_count: u32) -> Result<Vec<u32>> {
        let returned_slots = device_read::<u32>(
            "moonai_gpu_simulation_read_free_slot_count",
            match population_kind {
                PopulationKind::Predator => self.device_state.predator_free_len,
                PopulationKind::Prey => self.device_state.prey_free_len,
            },
        )?
        .min(slot_count);
        let returned_slots = usize::try_from(returned_slots).context("free-slot length overflowed")?;
        let mut slots = vec![0_u32; returned_slots];
        if returned_slots > 0 {
            device_read_slice(
                "moonai_gpu_simulation_read_free_slots",
                match population_kind {
                    PopulationKind::Predator => self.device_state.predator_free_list,
                    PopulationKind::Prey => self.device_state.prey_free_list,
                },
                &mut slots,
            )?;
        }
        slots.truncate(returned_slots);
        Ok(slots)
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

    fn crossover_slot(
        &mut self,
        population_kind: PopulationKind,
        parent_a_slot: u32,
        parent_b_slot: u32,
        offspring_slot: u32,
    ) -> Result<()> {
        let status = unsafe {
            dev_crossover(self.get_dev_state(), population_kind, parent_a_slot, parent_b_slot, offspring_slot)
        };
        check_cuda_status(status, "moonai_gpu_evolution_crossover")
    }

    fn mutate_slot(&mut self, population_kind: PopulationKind, slot: u32, config: GpuMutationConfig) -> Result<()> {
        let status = unsafe { dev_mutate_slot(self.get_dev_state(), population_kind, slot, &config) };
        check_cuda_status(status, "moonai_gpu_evolution_mutate_slot")
    }

    fn apply_reproduction_energy(&mut self, population_kind: PopulationKind, births_applied: u32) -> Result<()> {
        if births_applied > 0 {
            check_cuda_status(
                unsafe { dev_apply_reproduction_energy_kernel(self.get_dev_state(), population_kind, births_applied) },
                "moonai_gpu_simulation_apply_reproduction_energy",
            )?;
        }
        let status = unsafe { dev_initialize_population_free_list(self.get_dev_state(), population_kind) };
        check_cuda_status(status, "moonai_gpu_simulation_rebuild_free_list")
    }

    const unsafe fn get_dev_state(&mut self) -> *mut DeviceState {
        &mut self.device_state as *mut DeviceState
    }

    fn clean_spatial_grid_buffers(&mut self) -> Result<()> {
        cuda_free(&mut self.device_state.predator_cell_counts)?;
        cuda_free(&mut self.device_state.predator_cell_offsets)?;
        cuda_free(&mut self.device_state.predator_cell_write_offsets)?;
        cuda_free(&mut self.device_state.predator_grid_entries)?;
        cuda_free(&mut self.device_state.prey_cell_counts)?;
        cuda_free(&mut self.device_state.prey_cell_offsets)?;
        cuda_free(&mut self.device_state.prey_cell_write_offsets)?;
        cuda_free(&mut self.device_state.prey_grid_entries)?;
        cuda_free(&mut self.device_state.food_cell_counts)?;
        cuda_free(&mut self.device_state.food_cell_offsets)?;
        cuda_free(&mut self.device_state.food_cell_write_offsets)?;
        cuda_free(&mut self.device_state.food_grid_entries)?;
        self.device_state.grid_cols = 0;
        self.device_state.grid_rows = 0;
        self.device_state.grid_cell_capacity = 0;
        self.device_state.grid_cell_size = 0.0;

        Ok(())
    }

    fn free_reproduction_buffers(&mut self) -> Result<()> {
        cuda_free(&mut self.device_state.predator_mate_claims)?;
        cuda_free(&mut self.device_state.prey_mate_claims)?;
        cuda_free(&mut self.device_state.predator_reproduction_pairs)?;
        cuda_free(&mut self.device_state.prey_reproduction_pairs)?;
        cuda_free(&mut self.device_state.predator_pair_count)?;
        cuda_free(&mut self.device_state.prey_pair_count)?;

        Ok(())
    }

    fn clean_buffers(&mut self) -> Result<()> {
        self.device_state.food.clean_buffers()?;

        self.device_state.predator.clean_buffers()?;
        self.device_state.prey.clean_buffers()?;
        cuda_free(&mut self.device_state.innovation)?;
        cuda_free(&mut self.device_state.next_entity_id)?;
        cuda_free(&mut self.device_state.counters)?;
        self.free_free_list_buffers()?;
        self.free_reproduction_buffers()?;
        cuda_free(&mut self.device_state.food_claimed_by)?;
        cuda_free(&mut self.device_state.population_live_count_scratch)?;
        cuda_free(&mut self.device_state.ui_stats_scratch)?;
        cuda_free(&mut self.device_state.free_list_state_scratch)?;
        cuda_free(&mut self.device_state.sensor_snapshot_scratch)?;
        cuda_free(&mut self.device_state.compiled_header_scratch)?;
        cuda_free(&mut self.device_state.selected_network_scratch)?;
        cuda_free(&mut self.device_state.metrics_summary)?;
        cuda_free(&mut self.device_state.metrics_reduce_scratch)?;
        cuda_free(&mut self.device_state.species_summaries_scratch)?;
        cuda_free(&mut self.device_state.representative_headers_scratch)?;
        cuda_free(&mut self.device_state.species_count_scratch)?;
        cuda_free(&mut self.device_state.render_header_scratch)?;
        cuda_free(&mut self.device_state.render_predators_scratch)?;
        cuda_free(&mut self.device_state.render_prey_scratch)?;
        cuda_free(&mut self.device_state.render_food_scratch)?;
        self.clean_spatial_grid_buffers()?;

        Ok(())
    }
}

unsafe extern "C" {
    fn dev_seed_initial_population(state: *mut DeviceState) -> i32;
    fn dev_write_population_live_count(state: *mut DeviceState, population_kind: PopulationKind) -> i32;
    fn dev_reset_counters(state: *mut DeviceState) -> i32;
    fn dev_initialize_population_free_list(state: *mut DeviceState, population_kind: PopulationKind) -> i32;
    fn dev_seed_food(state: *mut DeviceState) -> i32;
    fn dev_reset_population_reproduction_state(state: *mut DeviceState, population_kind: PopulationKind) -> i32;
    fn dev_count_population_cells(state: *mut DeviceState, population_kind: PopulationKind) -> i32;
    fn dev_count_food_cells(state: *mut DeviceState) -> i32;
    fn dev_exclusive_scan_u32(input: *mut u32, count: u32, output: *mut u32) -> i32;
    fn dev_finalize_population_cell_offsets(
        state: *mut DeviceState,
        population_kind: PopulationKind,
        cell_count: u32,
    ) -> i32;
    fn dev_finalize_food_cell_offsets(state: *mut DeviceState, cell_count: u32) -> i32;
    fn dev_scatter_population_cells(state: *mut DeviceState, population_kind: PopulationKind) -> i32;
    fn dev_scatter_food_cells(state: *mut DeviceState) -> i32;
    fn dev_compute_sensor_inputs(state: *mut DeviceState) -> i32;
    fn dev_infer_population(state: *mut DeviceState, population_kind: PopulationKind) -> i32;
    fn dev_update_vitals(state: *mut DeviceState, population_kind: PopulationKind) -> i32;
    fn dev_resolve_food_claims(state: *mut DeviceState) -> i32;
    fn dev_finalize_food(state: *mut DeviceState) -> i32;
    fn dev_respawn_food(state: *mut DeviceState) -> i32;
    fn dev_resolve_combat_claims(state: *mut DeviceState) -> i32;
    fn dev_finalize_combat(state: *mut DeviceState) -> i32;
    fn dev_apply_movement(state: *mut DeviceState, population_kind: PopulationKind) -> i32;
    fn dev_find_reproduction_pairs(state: *mut DeviceState, population_kind: PopulationKind) -> i32;
    fn dev_apply_reproduction_energy_kernel(
        state: *mut DeviceState,
        population_kind: PopulationKind,
        births_applied: u32,
    ) -> i32;
    fn dev_advance_tick(state: *mut DeviceState) -> i32;
    fn dev_write_ui_stats(state: *mut DeviceState) -> i32;
    fn dev_write_free_list_state(state: *mut DeviceState) -> i32;
    fn dev_accumulate_metrics(state: *mut DeviceState, population_kind: PopulationKind) -> i32;
    fn dev_finalize_metrics_summary(state: *mut DeviceState) -> i32;
    fn dev_write_sensor_snapshot(state: *mut DeviceState, population_kind: PopulationKind, slot: u32) -> i32;
    fn dev_initialize_render_snapshot(state: *mut DeviceState) -> i32;
    fn dev_pack_render_agents(state: *mut DeviceState, population_kind: PopulationKind, max_count: u32) -> i32;
    fn dev_pack_render_food(state: *mut DeviceState, max_food: u32) -> i32;
    fn dev_compile_population(state: *mut DeviceState, population_kind: PopulationKind) -> i32;
    fn dev_compile_slot(state: *mut DeviceState, population_kind: PopulationKind, slot: u32) -> i32;
    fn dev_write_compiled_header(state: *mut DeviceState, population_kind: PopulationKind, slot: u32) -> i32;
    fn dev_crossover(
        state: *mut DeviceState,
        population_kind: PopulationKind,
        parent_a_slot: u32,
        parent_b_slot: u32,
        offspring_slot: u32,
    ) -> i32;
    fn dev_mutate_slot(
        state: *mut DeviceState,
        population_kind: PopulationKind,
        slot: u32,
        config: *const GpuMutationConfig,
    ) -> i32;
    fn dev_write_selected_agent_network(state: *mut DeviceState, population_kind: PopulationKind, slot: u32) -> i32;
    fn dev_classify_species_summaries(state: *mut DeviceState, population_kind: PopulationKind) -> i32;
}
