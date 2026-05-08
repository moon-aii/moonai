use crate::tick::buffers::{RenderAgentReadback, RenderSnapshotReadback, UiStatsReadback};
use crate::tick::genome::PopulationKind;
use crate::tick::inference::SensorSnapshotReadback;
use crate::tick::network::SelectedAgentNetworkReadback;
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
    pub speed_multiplier: u32,
    pub paused: bool,
    pub fps: f32,
    pub predators_returned: usize,
    pub prey_returned: usize,
    pub food_returned: usize,
}

impl OverlayStats {
    pub const fn from_snapshot(
        snapshot: &RenderSnapshotReadback,
        ui_stats: UiStatsReadback,
        speed_multiplier: u32,
        paused: bool,
        fps: f32,
    ) -> Self {
        Self {
            ui_stats,
            speed_multiplier,
            paused,
            fps,
            predators_returned: snapshot.predators.len(),
            prey_returned: snapshot.prey.len(),
            food_returned: snapshot.food.len(),
        }
    }
}
