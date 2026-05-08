use std::collections::VecDeque;

use crate::tick::buffers::{MetricsSummaryReadback, RenderAgentReadback, UiStatsReadback};
use crate::tick::network::SelectedAgentNetworkReadback;
use crate::tick::network::SensorSnapshotReadback;
use crate::tick::simulation::PopulationKind;
use crate::tick::species::RepresentativeGenomeReadback;

#[derive(Debug, Clone)]
pub struct UiState {
    pub paused: bool,
    pub tick_requested: bool,
    pub speed_multiplier: u32,
    pub selected_agent_id: Option<u32>,
}

impl Default for UiState {
    fn default() -> Self {
        Self { paused: false, tick_requested: false, speed_multiplier: 1, selected_agent_id: None }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraState {
    pub center_x: f32,
    pub center_y: f32,
    pub zoom: f32,
}

impl CameraState {
    pub const fn new(center_x: f32, center_y: f32, zoom: f32) -> Self {
        Self { center_x, center_y, zoom }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelectedAgent {
    pub population_kind: PopulationKind,
    pub slot: u32,
    pub entity_id: u32,
}

impl SelectedAgent {
    pub const fn from_render_agent(agent: &RenderAgentReadback) -> Self {
        Self { population_kind: agent.population_kind, slot: agent.slot, entity_id: agent.entity_id }
    }
}

#[derive(Debug, Clone)]
pub struct SelectedAgentData {
    pub agent: RenderAgentReadback,
    pub sensors: SensorSnapshotReadback,
    pub network: SelectedAgentNetworkReadback,
    pub genome: RepresentativeGenomeReadback,
}

#[derive(Debug, Clone)]
pub struct OverlayStats {
    pub ui_stats: UiStatsReadback,
    pub metrics_summary: MetricsSummaryReadback,
    pub speed_multiplier: u32,
    pub paused: bool,
    pub fps: f32,
    pub active_food_count: u32,
}

impl OverlayStats {
    pub const fn from_snapshot(
        ui_stats: UiStatsReadback,
        metrics_summary: MetricsSummaryReadback,
        active_food_count: u32,
        speed_multiplier: u32,
        paused: bool,
        fps: f32,
    ) -> Self {
        Self { ui_stats, metrics_summary, speed_multiplier, paused, fps, active_food_count }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PopulationHistoryPoint {
    pub predators: u32,
    pub prey: u32,
    pub food: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct PairHistoryPoint {
    pub predator: f32,
    pub prey: f32,
}

#[derive(Debug, Default, Clone)]
pub struct OverlayHistory {
    pub last_tick: Option<u32>,
    pub population: VecDeque<PopulationHistoryPoint>,
    pub complexity: VecDeque<PairHistoryPoint>,
    pub energy: VecDeque<PairHistoryPoint>,
}

impl OverlayHistory {
    pub fn push(&mut self, overlay: &OverlayStats) {
        if self.last_tick == Some(overlay.ui_stats.tick) {
            return;
        }

        self.last_tick = Some(overlay.ui_stats.tick);
        self.population.push_back(PopulationHistoryPoint {
            predators: overlay.ui_stats.predator_count,
            prey: overlay.ui_stats.prey_count,
            food: overlay.active_food_count,
        });
        self.complexity.push_back(PairHistoryPoint {
            predator: overlay.metrics_summary.avg_predator_complexity,
            prey: overlay.metrics_summary.avg_prey_complexity,
        });
        self.energy.push_back(PairHistoryPoint {
            predator: overlay.ui_stats.avg_predator_energy,
            prey: overlay.ui_stats.avg_prey_energy,
        });
    }
}
