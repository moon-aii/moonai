use mlua::Lua;
use std::collections::HashMap;

use crate::config::{ConfigError, SimulationConfig};

pub fn load_config(path: &str) -> Result<HashMap<String, SimulationConfig>, ConfigError> {
    let lua = Lua::new();
    let defaults = SimulationConfig::default();
    let globals = lua.globals();
    let moonai_defaults = lua.create_table().map_err(|e| ConfigError::LuaParse(e.to_string()))?;

    moonai_defaults.set("grid_size", defaults.grid_size as f64).map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("predator_count", defaults.predator_count as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults.set("prey_count", defaults.prey_count as f64).map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults.set("food_count", defaults.food_count as f64).map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("predator_speed", defaults.predator_speed as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults.set("prey_speed", defaults.prey_speed as f64).map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("vision_range", defaults.vision_range as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("interaction_range", defaults.interaction_range as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults.set("mate_range", defaults.mate_range as f64).map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("food_respawn_rate", defaults.food_respawn_rate as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("energy_drain_per_tick", defaults.energy_drain_per_tick as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("energy_gain_from_kill", defaults.energy_gain_from_kill as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("energy_gain_from_food", defaults.energy_gain_from_food as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("initial_energy", defaults.initial_energy as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults.set("max_energy", defaults.max_energy as f64).map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("reproduction_energy_threshold", defaults.reproduction_energy_threshold as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("reproduction_energy_cost", defaults.reproduction_energy_cost as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("offspring_initial_energy", defaults.offspring_initial_energy as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults.set("max_age", defaults.max_age as f64).map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("mutation_rate", defaults.mutation_rate as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("weight_mutation_power", defaults.weight_mutation_power as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("add_node_rate", defaults.add_node_rate as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("add_connection_rate", defaults.add_connection_rate as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("delete_connection_rate", defaults.delete_connection_rate as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("max_hidden_nodes", defaults.max_hidden_nodes as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults.set("max_ticks", defaults.max_ticks as f64).map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("compatibility_threshold", defaults.compatibility_threshold as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("compatibility_min_normalization", defaults.compatibility_min_normalization as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults.set("c1_excess", defaults.c1_excess as f64).map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("c2_disjoint", defaults.c2_disjoint as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults.set("c3_weight", defaults.c3_weight as f64).map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults.set("seed", defaults.seed as f64).map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("report_interval_ticks", defaults.report_interval_ticks as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;

    globals.set("moonai_defaults", moonai_defaults).map_err(|e| ConfigError::LuaParse(e.to_string()))?;

    let content = std::fs::read_to_string(path).map_err(ConfigError::Io)?;
    let experiments_table: mlua::Table = lua.load(&content).eval().map_err(|e| ConfigError::LuaParse(e.to_string()))?;

    let mut experiments = HashMap::new();
    for pair in experiments_table.pairs::<String, mlua::Table>() {
        let (name, cfg_table) = pair.map_err(|e| ConfigError::LuaParse(e.to_string()))?;
        let config = table_to_config(&cfg_table)?;
        experiments.insert(name, config);
    }
    Ok(experiments)
}

fn table_to_config(table: &mlua::Table) -> Result<SimulationConfig, ConfigError> {
    let mut config = SimulationConfig::default();
    macro_rules! set_i32 {
        ($k:literal, $f:ident) => {
            if let Ok(v) = table.get::<f64>($k) {
                config.$f = v as i32;
            }
        };
    }
    macro_rules! set_f32 {
        ($k:literal, $f:ident) => {
            if let Ok(v) = table.get::<f64>($k) {
                config.$f = v as f32;
            }
        };
    }
    set_i32!("grid_size", grid_size);
    set_i32!("predator_count", predator_count);
    set_i32!("prey_count", prey_count);
    set_i32!("food_count", food_count);
    set_f32!("predator_speed", predator_speed);
    set_f32!("prey_speed", prey_speed);
    set_f32!("vision_range", vision_range);
    set_f32!("interaction_range", interaction_range);
    set_f32!("mate_range", mate_range);
    set_f32!("food_respawn_rate", food_respawn_rate);
    set_f32!("energy_drain_per_tick", energy_drain_per_tick);
    set_f32!("energy_gain_from_kill", energy_gain_from_kill);
    set_f32!("energy_gain_from_food", energy_gain_from_food);
    set_f32!("initial_energy", initial_energy);
    set_f32!("max_energy", max_energy);
    set_f32!("reproduction_energy_threshold", reproduction_energy_threshold);
    set_f32!("reproduction_energy_cost", reproduction_energy_cost);
    set_f32!("offspring_initial_energy", offspring_initial_energy);
    set_i32!("max_age", max_age);
    set_f32!("mutation_rate", mutation_rate);
    set_f32!("weight_mutation_power", weight_mutation_power);
    set_f32!("add_node_rate", add_node_rate);
    set_f32!("add_connection_rate", add_connection_rate);
    set_f32!("delete_connection_rate", delete_connection_rate);
    set_i32!("max_hidden_nodes", max_hidden_nodes);
    set_i32!("max_ticks", max_ticks);
    set_f32!("compatibility_threshold", compatibility_threshold);
    set_f32!("compatibility_min_normalization", compatibility_min_normalization);
    set_f32!("c1_excess", c1_excess);
    set_f32!("c2_disjoint", c2_disjoint);
    set_f32!("c3_weight", c3_weight);
    set_i32!("seed", seed);
    set_i32!("report_interval_ticks", report_interval_ticks);
    Ok(config)
}
