mod cli;
mod config;
mod config_error;
mod lua;
mod metrics;
mod settings;
mod signal;
mod tick;
mod types;
mod ui;

use std::collections::HashMap;
use std::io::{self, Write as _};

use anyhow::Result;
use clap::Parser as _;

use crate::cli::CliArgs;
use crate::config::SimulationConfig;
use crate::config_error::{ConfigError, validate_config};

fn stdout_line(message: &str) {
    let mut stdout = io::stdout().lock();
    let _ = stdout.write_all(message.as_bytes());
    let _ = stdout.write_all(b"\n");
}

fn stderr_line(message: &str) {
    let mut stderr = io::stderr().lock();
    let _ = stderr.write_all(message.as_bytes());
    let _ = stderr.write_all(b"\n");
}

fn resolve_config_path(args: &CliArgs) -> Option<String> {
    args.config.as_ref().map_or_else(
        || settings::config_path_from_binary().map(|path| path.to_string_lossy().into_owned()),
        |path| Some(path.clone()),
    )
}

fn load_experiments(config_path: &str) -> Result<HashMap<String, SimulationConfig>, ConfigError> {
    lua::load_config(config_path)
}

fn default_or_only_experiment(experiments: &HashMap<String, SimulationConfig>) -> Option<SimulationConfig> {
    experiments
        .get("default")
        .cloned()
        .or_else(|| if experiments.len() == 1 { experiments.values().next().cloned() } else { None })
}

fn do_list(config_path: &str) -> Result<(), ConfigError> {
    let experiments = load_experiments(config_path)?;
    let mut names: Vec<_> = experiments.keys().map(String::as_str).collect();
    names.sort_unstable();
    stdout_line("Available experiments:");
    for name in names {
        stdout_line(&format!("  {name}"));
    }
    Ok(())
}

fn do_validate(config_path: &str, experiment_name: Option<&str>) -> Result<(), ConfigError> {
    let experiments = load_experiments(config_path)?;

    let config = experiment_name.map_or_else(
        || {
            default_or_only_experiment(&experiments).unwrap_or_else(|| {
                stderr_line("Error: no 'default' experiment found and multiple experiments exist");
                std::process::exit(1);
            })
        },
        |name| {
            experiments.get(name).cloned().unwrap_or_else(|| {
                stderr_line(&format!("Error: experiment '{name}' not found"));
                std::process::exit(1);
            })
        },
    );

    match validate_config(&config) {
        Ok(()) => {
            stdout_line("Configuration is valid.");
            Ok(())
        }
        Err(ConfigError::InvalidConfig(msg)) => {
            stderr_line(&format!("Configuration is invalid: {msg}"));
            std::process::exit(1);
        }
        Err(e) => Err(e),
    }
}

fn select_experiment(
    experiments: &HashMap<String, SimulationConfig>,
    name: Option<&str>,
) -> Result<SimulationConfig, ConfigError> {
    name.map_or_else(
        || {
            default_or_only_experiment(experiments).ok_or_else(|| {
                ConfigError::InvalidConfig(
                    "no 'default' experiment found and multiple experiments exist without --experiment flag".to_owned(),
                )
            })
        },
        |n| {
            experiments.get(n).cloned().ok_or_else(|| ConfigError::InvalidConfig(format!("experiment '{n}' not found")))
        },
    )
}

fn main() -> Result<()> {
    let args = CliArgs::parse();

    let Some(config_path) = resolve_config_path(&args) else {
        stderr_line("Error: config.lua not found. Provide with --config or place next to binary.");
        std::process::exit(1);
    };

    if args.list {
        do_list(&config_path)?;
        return Ok(());
    }

    if args.validate {
        do_validate(&config_path, args.experiment.as_deref())?;
        return Ok(());
    }

    let experiments = load_experiments(&config_path)?;

    if args.all {
        if !args.headless {
            stderr_line("Error: --all requires --headless");
            std::process::exit(1);
        }
        let mut runs: Vec<_> = experiments.iter().collect();
        runs.sort_by_key(|(name, _)| *name);
        for (name, base_config) in runs {
            let config = args.ticks.map_or_else(
                || base_config.clone(),
                |ticks| {
                    let mut cfg = base_config.clone();
                    cfg.max_ticks = ticks;
                    cfg
                },
            );
            stdout_line(&format!("Running experiment: {name}"));
            stdout_line(&format!("Config: {config:?}"));
        }
        stdout_line("All experiments completed.");
        return Ok(());
    }

    let config = select_experiment(&experiments, args.experiment.as_deref())?;
    let mut config = config;
    if let Some(ticks) = args.ticks {
        config.max_ticks = ticks;
    }

    stdout_line("MoonAI - GPU-first predator-prey evolution simulation");
    stdout_line(&format!("Loaded experiment: {:?}", args.experiment.as_deref().unwrap_or("default")));
    stdout_line(&format!("Config: {config:?}"));

    Ok(())
}
