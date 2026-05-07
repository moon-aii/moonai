use thiserror::Error;

use crate::config::SimulationConfig;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Lua parsing error: {0}")]
    LuaParse(String),
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Settings JSON error: {0}")]
    SettingsJson(#[from] serde_json::Error),
}

pub fn validate_config(config: &SimulationConfig) -> Result<(), ConfigError> {
    let mut errors = Vec::new();
    if config.grid_size < 100 {
        errors.push(format!("grid_size must be >= 100, got {}", config.grid_size));
    }
    if config.grid_size > 20000 {
        errors.push(format!("grid_size must be <= 20000, got {}", config.grid_size));
    }
    if config.predator_count < 1 {
        errors.push(format!("predator_count must be >= 1, got {}", config.predator_count));
    }
    if config.prey_count < 1 {
        errors.push(format!("prey_count must be >= 1, got {}", config.prey_count));
    }
    if config.predator_count + config.prey_count > 1_000_000 {
        errors.push(format!("total population must be <= 1000000, got {}", config.predator_count + config.prey_count));
    }
    if config.predator_speed <= 0.0 {
        errors.push(format!("predator_speed must be > 0, got {}", config.predator_speed));
    }
    if config.prey_speed <= 0.0 {
        errors.push(format!("prey_speed must be > 0, got {}", config.prey_speed));
    }
    if config.vision_range <= 0.0 {
        errors.push(format!("vision_range must be > 0, got {}", config.vision_range));
    }
    if config.interaction_range <= 0.0 {
        errors.push(format!("interaction_range must be > 0, got {}", config.interaction_range));
    }
    if config.interaction_range >= config.vision_range {
        errors.push(format!(
            "interaction_range must be < vision_range, got {} >= {}",
            config.interaction_range, config.vision_range
        ));
    }
    if config.initial_energy <= 0.0 {
        errors.push(format!("initial_energy must be > 0, got {}", config.initial_energy));
    }
    if config.max_energy <= 0.0 {
        errors.push(format!("max_energy must be > 0, got {}", config.max_energy));
    }
    if config.initial_energy > config.max_energy {
        errors.push(format!(
            "initial_energy must be <= max_energy, got {} > {}",
            config.initial_energy, config.max_energy
        ));
    }
    if config.energy_drain_per_tick < 0.0 {
        errors.push(format!("energy_drain_per_tick must be >= 0, got {}", config.energy_drain_per_tick));
    }
    if config.food_count < 0 {
        errors.push(format!("food_count must be >= 0, got {}", config.food_count));
    }
    if !(0.0..=1.0).contains(&config.food_respawn_rate) {
        errors.push(format!("food_respawn_rate must be in [0, 1], got {}", config.food_respawn_rate));
    }
    if !(0.0..=1.0).contains(&config.mutation_rate) {
        errors.push(format!("mutation_rate must be in [0, 1], got {}", config.mutation_rate));
    }
    if !(0.0..=1.0).contains(&config.add_node_rate) {
        errors.push(format!("add_node_rate must be in [0, 1], got {}", config.add_node_rate));
    }
    if !(0.0..=1.0).contains(&config.add_connection_rate) {
        errors.push(format!("add_connection_rate must be in [0, 1], got {}", config.add_connection_rate));
    }
    if !(0.0..=1.0).contains(&config.delete_connection_rate) {
        errors.push(format!("delete_connection_rate must be in [0, 1], got {}", config.delete_connection_rate));
    }
    if config.weight_mutation_power <= 0.0 {
        errors.push(format!("weight_mutation_power must be > 0, got {}", config.weight_mutation_power));
    }
    if config.max_age < 0 {
        errors.push(format!("max_age must be >= 0 (0 = infinite), got {}", config.max_age));
    }
    if config.max_ticks < 0 {
        errors.push(format!("max_ticks must be >= 0 (0 = infinite), got {}", config.max_ticks));
    }
    if config.compatibility_threshold <= 0.0 {
        errors.push(format!("compatibility_threshold must be > 0, got {}", config.compatibility_threshold));
    }
    if config.compatibility_min_normalization < 1.0 {
        errors.push(format!(
            "compatibility_min_normalization must be >= 1, got {}",
            config.compatibility_min_normalization
        ));
    }
    if config.report_interval_ticks < 1 {
        errors.push(format!("report_interval_ticks must be >= 1, got {}", config.report_interval_ticks));
    }
    if config.mate_range <= 0.0 {
        errors.push(format!("mate_range must be > 0, got {}", config.mate_range));
    }
    if config.reproduction_energy_threshold <= 0.0 {
        errors.push(format!("reproduction_energy_threshold must be > 0, got {}", config.reproduction_energy_threshold));
    }
    if config.reproduction_energy_threshold > config.max_energy {
        errors.push(format!(
            "reproduction_energy_threshold must be <= max_energy, got {} > {}",
            config.reproduction_energy_threshold, config.max_energy
        ));
    }
    if config.reproduction_energy_cost <= 0.0 {
        errors.push(format!("reproduction_energy_cost must be > 0, got {}", config.reproduction_energy_cost));
    }
    if config.offspring_initial_energy <= 0.0 {
        errors.push(format!("offspring_initial_energy must be > 0, got {}", config.offspring_initial_energy));
    }
    if config.offspring_initial_energy > config.max_energy {
        errors.push(format!(
            "offspring_initial_energy must be <= max_energy, got {} > {}",
            config.offspring_initial_energy, config.max_energy
        ));
    }

    if errors.is_empty() { Ok(()) } else { Err(ConfigError::InvalidConfig(errors.join("; "))) }
}
