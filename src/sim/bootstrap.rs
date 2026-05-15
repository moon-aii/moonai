use super::*;

use anyhow::{Context as _, Result};
use std::ptr;

impl Simulation {
    pub(super) fn allocate_core_state_buffers(&mut self) -> Result<()> {
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
        cuda_malloc(&mut self.device_state.ui_stats_reduce_scratch, 1)?;
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

    pub(super) fn allocate_population_render_buffers(&mut self) -> Result<()> {
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

    pub(super) fn allocate_food_state_buffers(&mut self) -> Result<()> {
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

    pub(super) fn allocate_counter_buffer(&mut self) -> Result<()> {
        cuda_malloc(&mut self.device_state.counters, 1)
    }

    pub(super) fn free_free_list_buffers(&mut self) -> Result<()> {
        cuda_free(&mut self.device_state.predator_free_list)?;
        cuda_free(&mut self.device_state.prey_free_list)?;
        cuda_free(&mut self.device_state.predator_free_len)?;
        cuda_free(&mut self.device_state.prey_free_len)?;
        cuda_free(&mut self.device_state.prey_claimed_by)?;

        Ok(())
    }

    pub(super) fn allocate_free_list_buffers(&mut self) -> Result<()> {
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

    pub(super) fn initialize_device_scalars(&mut self) -> Result<()> {
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

    pub(super) fn init_dev(&mut self) -> Result<()> {
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
        self.reset_reproduction_state(PopulationKind::Predator).context("dev_reset_predator_reproduction_state")?;
        self.reset_reproduction_state(PopulationKind::Prey).context("dev_reset_prey_reproduction_state")?;

        if self.device_state.predator.capacity > 0 {
            let _ = self.compile_population(PopulationKind::Predator, 0)?;
        }
        if self.device_state.prey.capacity > 0 {
            let _ = self.compile_population(PopulationKind::Prey, 0)?;
        }

        self.refresh_reports()?;

        Ok(())
    }

    pub(super) fn clean_buffers(&mut self) -> Result<()> {
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
        cuda_free(&mut self.device_state.ui_stats_reduce_scratch)?;
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
    fn dev_reset_counters(state: *mut DeviceState) -> i32;
    fn dev_initialize_population_free_list(state: *mut DeviceState, population_kind: PopulationKind) -> i32;
    fn dev_seed_food(state: *mut DeviceState) -> i32;
}
