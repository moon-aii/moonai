use super::*;

use anyhow::{Context as _, Result};

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

    pub(super) fn ensure_birth_capacity(&mut self, population_kind: PopulationKind, births_pending: u32) -> Result<()> {
        profile_scope!("birth_cap");

        if births_pending == 0 {
            return Ok(());
        }

        let live_count = self.population_live_count(population_kind)?;
        let free_list_state = self.read_free_list_state()?;
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

    pub(super) fn run_reproduction(&mut self, population_kind: PopulationKind) -> Result<()> {
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
}

unsafe extern "C" {
    fn dev_reset_population_reproduction_state(state: *mut DeviceState, population_kind: PopulationKind) -> i32;
    fn dev_find_reproduction_pairs(state: *mut DeviceState, population_kind: PopulationKind) -> i32;
    fn dev_apply_reproduction_energy_kernel(
        state: *mut DeviceState,
        population_kind: PopulationKind,
        births_applied: u32,
    ) -> i32;
    fn dev_initialize_population_free_list(state: *mut DeviceState, population_kind: PopulationKind) -> i32;
}
