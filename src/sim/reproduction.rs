use super::*;

use anyhow::Result;

impl Simulation {
    pub(super) fn allocate_reproduction_buffers(&mut self) -> Result<()> {
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

    pub(super) fn free_reproduction_buffers(&mut self) -> Result<()> {
        cuda_free(&mut self.device_state.predator_mate_claims)?;
        cuda_free(&mut self.device_state.prey_mate_claims)?;
        cuda_free(&mut self.device_state.predator_reproduction_pairs)?;
        cuda_free(&mut self.device_state.prey_reproduction_pairs)?;
        cuda_free(&mut self.device_state.predator_pair_count)?;
        cuda_free(&mut self.device_state.prey_pair_count)?;

        Ok(())
    }

    pub(super) fn ensure_birth_capacity(
        &mut self,
        population_kind: PopulationKind,
        births_pending: u32,
    ) -> Result<bool> {
        profile_scope!("birth_cap");

        if births_pending == 0 {
            return Ok(false);
        }

        let free_slots = self.population_free_slot_count(population_kind)?;
        let capacity = self.population_capacity(population_kind);
        let live_count = capacity.saturating_sub(free_slots);
        let required_live = live_count.saturating_add(births_pending);
        if free_slots >= births_pending && required_live <= ((capacity * 9) / 10) {
            return Ok(false);
        }

        let mut new_capacity = if capacity == 0 { 1 } else { capacity };
        while new_capacity.saturating_sub(live_count) < births_pending || required_live > ((new_capacity * 9) / 10) {
            new_capacity = if new_capacity == 0 { 1 } else { new_capacity.saturating_mul(2) };
        }
        self.expand_population(population_kind, new_capacity)?;
        self.build_spatial_grid()?;
        Ok(true)
    }

    fn population_free_slot_count(&mut self, population_kind: PopulationKind) -> Result<u32> {
        device_read(
            "moonai_gpu_simulation_population_free_slot_count",
            match population_kind {
                PopulationKind::Predator => self.device_state.predator_free_len,
                PopulationKind::Prey => self.device_state.prey_free_len,
            },
        )
    }

    pub(super) fn reset_reproduction_state(&mut self, population_kind: PopulationKind) -> Result<()> {
        check_cuda_status(
            unsafe { dev_reset_population_reproduction_state(self.get_dev_state(), population_kind) },
            "moonai_gpu_simulation_reset_reproduction_state",
        )
    }

    pub(super) fn reproduction_candidate_count(&mut self, population_kind: PopulationKind) -> Result<u32> {
        profile_scope!("reprod_candidate");
        self.reset_reproduction_state(population_kind)?;
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

    pub(super) fn expand_population(&mut self, population_kind: PopulationKind, new_capacity: u32) -> Result<()> {
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

    pub(super) fn run_reproduction(&mut self, population_kind: PopulationKind, pair_count: u32) -> Result<()> {
        profile_scope!("run_reprod");
        if pair_count == 0 {
            return Ok(());
        }

        let free_slots = self.population_free_slot_count(population_kind)?;
        let births_applied = pair_count.min(free_slots);
        if births_applied == 0 {
            return Ok(());
        }

        let mutation_config = self.mutation_config()?;
        let free_slot_base = free_slots.saturating_sub(births_applied);
        self.crossover_batch(population_kind, births_applied, free_slot_base)?;
        self.mutate_batch(population_kind, births_applied, free_slot_base, mutation_config)?;
        self.compile_slots_batch(population_kind, births_applied, free_slot_base)?;
        self.apply_reproduction_energy(population_kind, births_applied, free_slot_base)?;
        cuda_synchronize("moonai_gpu_simulation_run_reproduction")
    }

    fn apply_reproduction_energy(
        &mut self,
        population_kind: PopulationKind,
        births_applied: u32,
        free_slot_base: u32,
    ) -> Result<()> {
        check_cuda_status(
            unsafe {
                dev_apply_reproduction_energy_kernel(
                    self.get_dev_state(),
                    population_kind,
                    births_applied,
                    free_slot_base,
                )
            },
            "moonai_gpu_simulation_apply_reproduction_energy",
        )
    }
}

unsafe extern "C" {
    fn dev_reset_population_reproduction_state(state: *mut DeviceState, population_kind: PopulationKind) -> i32;
    fn dev_find_reproduction_pairs(state: *mut DeviceState, population_kind: PopulationKind) -> i32;
    fn dev_apply_reproduction_energy_kernel(
        state: *mut DeviceState,
        population_kind: PopulationKind,
        births_applied: u32,
        free_slot_base: u32,
    ) -> i32;
    fn dev_initialize_population_free_list(state: *mut DeviceState, population_kind: PopulationKind) -> i32;
}
