use super::*;

use anyhow::{Context as _, Result};

impl Simulation {
    pub(super) fn read_ui_stats(&mut self) -> Result<UiStatsReadback> {
        profile_scope!("ui_stats");
        let status = unsafe { dev_write_ui_stats(self.get_dev_state()) };
        check_cuda_status(status, "moonai_gpu_simulation_ui_stats")?;
        device_read("moonai_gpu_simulation_ui_stats_readback", self.device_state.ui_stats_scratch)
    }

    pub(super) fn read_free_list_state(&mut self) -> Result<FreeListStateReadback> {
        let status = unsafe { dev_write_free_list_state(self.get_dev_state()) };
        check_cuda_status(status, "moonai_gpu_simulation_free_list_state")?;
        device_read("moonai_gpu_simulation_free_list_state_readback", self.device_state.free_list_state_scratch)
    }

    pub(super) fn read_metrics_summary(&mut self) -> Result<MetricsSummaryReadback> {
        device_read("moonai_gpu_simulation_metrics_summary", self.device_state.metrics_summary)
    }

    pub(super) fn refresh_reports_impl(&mut self) -> Result<()> {
        profile_scope!("refresh_reports");
        self.classify_species_summaries_impl(PopulationKind::Predator)
            .context("moonai_gpu_simulation_classify_predator_species")?;
        self.classify_species_summaries_impl(PopulationKind::Prey)
            .context("moonai_gpu_simulation_classify_prey_species")?;
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

    pub(super) fn population_live_count(&mut self, population_kind: PopulationKind) -> Result<u32> {
        let status = unsafe { dev_write_population_live_count(self.get_dev_state(), population_kind) };
        check_cuda_status(status, "moonai_gpu_evolution_population_live_count")?;
        device_read(
            "moonai_gpu_evolution_population_live_count_readback",
            self.device_state.population_live_count_scratch,
        )
    }
}

unsafe extern "C" {
    fn dev_write_population_live_count(state: *mut DeviceState, population_kind: PopulationKind) -> i32;
    fn dev_write_ui_stats(state: *mut DeviceState) -> i32;
    fn dev_write_free_list_state(state: *mut DeviceState) -> i32;
    fn dev_accumulate_metrics(state: *mut DeviceState, population_kind: PopulationKind) -> i32;
    fn dev_finalize_metrics_summary(state: *mut DeviceState) -> i32;
}
