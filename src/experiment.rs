use mlua::Lua;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use thiserror::Error;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub grid_size: f32,
    pub predator_count: u32,
    pub prey_count: u32,
    pub food_count: u32,
    pub predator_speed: f32,
    pub prey_speed: f32,
    pub vision_range: f32,
    pub interaction_range: f32,
    pub mate_range: f32,
    pub food_respawn_rate: f32,
    pub energy_drain_per_tick: f32,
    pub energy_gain_from_kill: f32,
    pub energy_gain_from_food: f32,
    pub initial_energy: f32,
    pub max_energy: f32,
    pub reproduction_energy_threshold: f32,
    pub reproduction_energy_cost: f32,
    pub offspring_initial_energy: f32,
    pub max_age: u32,
    pub mutation_rate: f32,
    pub weight_mutation_power: f32,
    pub add_node_rate: f32,
    pub add_connection_rate: f32,
    pub delete_connection_rate: f32,
    pub max_hidden_nodes: u32,
    pub max_ticks: u32,
    pub compatibility_threshold: f32,
    pub compatibility_min_normalization: f32,
    pub c1_excess: f32,
    pub c2_disjoint: f32,
    pub c3_weight: f32,
    pub seed: u64,
    pub report_interval_ticks: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Experiment {
    pub name: String,
    pub path: PathBuf,
    pub simulation_config: SimulationConfig,
    pub is_default: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentCatalog {
    path: PathBuf,
    experiments: Vec<Experiment>,
    default_index: usize,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            grid_size: 3000.0,
            predator_count: 12000,
            prey_count: 48000,
            food_count: 60000,
            predator_speed: 1.0,
            prey_speed: 1.006,
            vision_range: 12.0,
            interaction_range: 1.0,
            mate_range: 6.0,
            food_respawn_rate: 0.01,
            energy_drain_per_tick: 0.001,
            energy_gain_from_kill: 0.24,
            energy_gain_from_food: 0.30,
            initial_energy: 0.36,
            max_energy: 2.0,
            reproduction_energy_threshold: 1.0,
            reproduction_energy_cost: 0.18,
            offspring_initial_energy: 0.36,
            max_age: 10000,
            mutation_rate: 0.30,
            weight_mutation_power: 0.30,
            add_node_rate: 0.12,
            add_connection_rate: 0.60,
            delete_connection_rate: 0.00000006,
            max_hidden_nodes: 1200,
            max_ticks: 0,
            compatibility_threshold: 60.0,
            compatibility_min_normalization: 240.0,
            c1_excess: 1.0,
            c2_disjoint: 1.0,
            c3_weight: 0.4,
            seed: 67,
            report_interval_ticks: 1000,
        }
    }
}

#[derive(Debug, Error)]
pub enum ExperimentError {
    #[error("Lua parsing error: {0}")]
    LuaParse(String),
    #[error("Invalid experiment configuration: {0}")]
    InvalidExperiment(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl ExperimentCatalog {
    pub fn load(root_dir: &Path) -> Result<Self, ExperimentError> {
        let path = root_dir.join("experiments.lua");
        let lua = Lua::new();
        let defaults = SimulationConfig::default();
        let globals = lua.globals();
        let moonai_defaults = lua.create_table().map_err(|e| ExperimentError::LuaParse(e.to_string()))?;

        moonai_defaults
            .set("grid_size", defaults.grid_size as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults
            .set("predator_count", defaults.predator_count as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults
            .set("prey_count", defaults.prey_count as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults
            .set("food_count", defaults.food_count as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults
            .set("predator_speed", defaults.predator_speed as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults
            .set("prey_speed", defaults.prey_speed as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults
            .set("vision_range", defaults.vision_range as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults
            .set("interaction_range", defaults.interaction_range as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults
            .set("mate_range", defaults.mate_range as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults
            .set("food_respawn_rate", defaults.food_respawn_rate as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults
            .set("energy_drain_per_tick", defaults.energy_drain_per_tick as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults
            .set("energy_gain_from_kill", defaults.energy_gain_from_kill as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults
            .set("energy_gain_from_food", defaults.energy_gain_from_food as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults
            .set("initial_energy", defaults.initial_energy as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults
            .set("max_energy", defaults.max_energy as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults
            .set("reproduction_energy_threshold", defaults.reproduction_energy_threshold as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults
            .set("reproduction_energy_cost", defaults.reproduction_energy_cost as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults
            .set("offspring_initial_energy", defaults.offspring_initial_energy as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults
            .set("max_age", defaults.max_age as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults
            .set("mutation_rate", defaults.mutation_rate as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults
            .set("weight_mutation_power", defaults.weight_mutation_power as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults
            .set("add_node_rate", defaults.add_node_rate as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults
            .set("add_connection_rate", defaults.add_connection_rate as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults
            .set("delete_connection_rate", defaults.delete_connection_rate as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults
            .set("max_hidden_nodes", defaults.max_hidden_nodes as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults
            .set("max_ticks", defaults.max_ticks as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults
            .set("compatibility_threshold", defaults.compatibility_threshold as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults
            .set("compatibility_min_normalization", defaults.compatibility_min_normalization as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults
            .set("c1_excess", defaults.c1_excess as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults
            .set("c2_disjoint", defaults.c2_disjoint as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults
            .set("c3_weight", defaults.c3_weight as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults.set("seed", defaults.seed as f64).map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
        moonai_defaults
            .set("report_interval_ticks", defaults.report_interval_ticks as f64)
            .map_err(|e| ExperimentError::LuaParse(e.to_string()))?;

        globals.set("moonai_defaults", moonai_defaults).map_err(|e| ExperimentError::LuaParse(e.to_string()))?;

        let content = std::fs::read_to_string(&path).map_err(ExperimentError::Io)?;
        let experiments_table: mlua::Table =
            lua.load(&content).eval().map_err(|e| ExperimentError::LuaParse(e.to_string()))?;

        let mut experiments = Vec::new();
        for pair in experiments_table.pairs::<String, mlua::Table>() {
            let (name, cfg_table) = pair.map_err(|e| ExperimentError::LuaParse(e.to_string()))?;
            let simulation_config = table_to_config(&cfg_table)?;
            validate_config(&simulation_config)?;
            experiments.push(Experiment { is_default: name == "default", name, path: path.clone(), simulation_config });
        }

        experiments.sort_by(|left, right| left.name.cmp(&right.name));
        if experiments.is_empty() {
            return Err(ExperimentError::InvalidExperiment(
                "experiments.lua did not define any experiments".to_owned(),
            ));
        }

        let default_index = experiments.iter().position(|experiment| experiment.is_default).unwrap_or(0);
        Ok(Self { path, experiments, default_index })
    }

    pub fn experiments(&self) -> &[Experiment] {
        &self.experiments
    }

    pub fn default(&self) -> &Experiment {
        &self.experiments[self.default_index]
    }

    pub fn get(&self, index: usize) -> Option<&Experiment> {
        self.experiments.get(index)
    }

    pub const fn default_index(&self) -> usize {
        self.default_index
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub const fn len(&self) -> usize {
        self.experiments.len()
    }

    pub const fn is_empty(&self) -> bool {
        self.experiments.is_empty()
    }
}

pub fn validate_config(config: &SimulationConfig) -> Result<(), ExperimentError> {
    let mut errors = Vec::new();
    if config.grid_size < 1.0 {
        errors.push(format!("grid_size must be >= 1, got {}", config.grid_size));
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

    if errors.is_empty() { Ok(()) } else { Err(ExperimentError::InvalidExperiment(errors.join("; "))) }
}

fn table_to_config(table: &mlua::Table) -> Result<SimulationConfig, ExperimentError> {
    let mut config = SimulationConfig::default();
    macro_rules! set_f32 {
        ($k:literal, $f:ident) => {
            if let Ok(v) = table.get::<f64>($k) {
                config.$f = v as f32;
            }
        };
    }
    macro_rules! set_u64 {
        ($k:literal, $f:ident) => {
            if let Ok(v) = table.get::<f64>($k) {
                if v < 0.0 {
                    return Err(ExperimentError::InvalidExperiment(format!("{} must be >= 0, got {}", $k, v)));
                }
                config.$f = v as u64;
            }
        };
    }
    macro_rules! set_u32 {
        ($k:literal, $f:ident) => {
            if let Ok(v) = table.get::<f64>($k) {
                if v < 0.0 {
                    return Err(ExperimentError::InvalidExperiment(format!("{} must be >= 0, got {}", $k, v)));
                }
                config.$f = v as u32;
            }
        };
    }
    set_f32!("grid_size", grid_size);
    set_u32!("predator_count", predator_count);
    set_u32!("prey_count", prey_count);
    set_u32!("food_count", food_count);
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
    set_u32!("max_age", max_age);
    set_f32!("mutation_rate", mutation_rate);
    set_f32!("weight_mutation_power", weight_mutation_power);
    set_f32!("add_node_rate", add_node_rate);
    set_f32!("add_connection_rate", add_connection_rate);
    set_f32!("delete_connection_rate", delete_connection_rate);
    set_u32!("max_hidden_nodes", max_hidden_nodes);
    set_u32!("max_ticks", max_ticks);
    set_f32!("compatibility_threshold", compatibility_threshold);
    set_f32!("compatibility_min_normalization", compatibility_min_normalization);
    set_f32!("c1_excess", c1_excess);
    set_f32!("c2_disjoint", c2_disjoint);
    set_f32!("c3_weight", c3_weight);
    set_u64!("seed", seed);
    set_u32!("report_interval_ticks", report_interval_ticks);
    Ok(config)
}
