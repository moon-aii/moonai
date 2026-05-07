use crate::config::SimulationConfig;

pub struct SimulationState;

impl Default for SimulationState {
    fn default() -> Self {
        Self::new()
    }
}

impl SimulationState {
    pub const fn new() -> Self {
        Self
    }

    pub const fn init_from_config(_config: &SimulationConfig) -> anyhow::Result<Self> {
        Ok(Self)
    }
}
