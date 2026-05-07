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

pub struct OverlayStats;

pub struct RenderFood;
pub struct RenderAgent;
pub struct RenderLine;
