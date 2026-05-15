use super::*;

use anyhow::{Context as _, Result};

impl Simulation {
    pub(super) fn allocate_spatial_grid_buffers(&mut self) -> Result<()> {
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

    pub(super) fn clean_spatial_grid_buffers(&mut self) -> Result<()> {
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

    pub(super) fn build_spatial_grid(&mut self) -> Result<()> {
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
        check_cuda_status(status, "moonai_gpu_simulation_scatter_food_cells")?;
        cuda_synchronize("moonai_gpu_simulation_build_spatial_grid")
    }
}

unsafe extern "C" {
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
}
