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
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result, bail};
use clap::Parser as _;

use crate::cli::CliArgs;
use crate::config::SimulationConfig;
use crate::config_error::{ConfigError, validate_config};
use crate::metrics::Logger;
use crate::tick::checks::CudaStatus;
use crate::tick::genome::PopulationKind;
use crate::tick::simulation::SimulationState;
use crate::tick::species::MAX_SPECIES_SUMMARIES;

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

fn select_named_experiment(
    experiments: &HashMap<String, SimulationConfig>,
    name: Option<&str>,
) -> Result<(String, SimulationConfig), ConfigError> {
    name.map_or_else(
        || {
            experiments
                .get("default")
                .cloned()
                .map(|config| ("default".to_owned(), config))
                .or_else(|| {
                    if experiments.len() == 1 {
                        experiments
                            .iter()
                            .next()
                            .map(|(experiment_name, config)| (experiment_name.clone(), config.clone()))
                    } else {
                        None
                    }
                })
                .ok_or_else(|| {
                    ConfigError::InvalidConfig(
                        "no 'default' experiment found and multiple experiments exist without --experiment flag"
                            .to_owned(),
                    )
                })
        },
        |n| {
            experiments
                .get(n)
                .cloned()
                .map(|config| (n.to_owned(), config))
                .ok_or_else(|| ConfigError::InvalidConfig(format!("experiment '{n}' not found")))
        },
    )
}

fn cuda_runtime_ready() -> Result<()> {
    let status = crate::tick::evolution::EvolutionManager::runtime_status();
    if status == CudaStatus::Success { Ok(()) } else { bail!("CUDA runtime unavailable: {status:?}") }
}

fn resolve_run_name(experiment_name: &str, explicit_name: Option<&str>) -> String {
    explicit_name.map_or_else(|| experiment_name.to_owned(), ToOwned::to_owned)
}

fn anonymous_run_name(seed: i32) -> String {
    let seconds = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |duration| duration.as_secs());
    format!("run_{seconds}_seed{seed}")
}

fn resolve_run_dir(output_root: &str, run_name: Option<&str>, seed: i32) -> PathBuf {
    let resolved_name = run_name.map_or_else(|| anonymous_run_name(seed), ToOwned::to_owned);
    Path::new(output_root).join(resolved_name)
}

fn log_population_report(
    state: &mut SimulationState,
    logger: &mut Logger,
    tick: u32,
    population_kind: PopulationKind,
) -> Result<()> {
    let (_, summaries, representatives) = state.species_summaries(population_kind, MAX_SPECIES_SUMMARIES)?;
    logger.log_species(tick, &summaries)?;

    let genomes = representatives
        .iter()
        .map(|representative| state.representative_genome(population_kind, representative.slot))
        .collect::<Result<Vec<_>>>()?;
    logger.log_genomes(tick, &genomes)?;
    Ok(())
}

fn log_report_snapshot(state: &mut SimulationState, logger: &mut Logger) -> Result<u32> {
    state.refresh_reports()?;
    let summary = state.metrics_summary()?;
    logger.log_stats(&summary)?;
    log_population_report(state, logger, summary.tick, PopulationKind::Predator)?;
    log_population_report(state, logger, summary.tick, PopulationKind::Prey)?;
    logger.flush()?;
    Ok(summary.tick)
}

fn run_headless_experiment(run_label: &str, config: &SimulationConfig, run_dir: &Path) -> Result<()> {
    validate_config(config)?;
    cuda_runtime_ready()?;
    signal::setup_signal_handlers();

    let report_interval = u32::try_from(config.report_interval_ticks).with_context(|| {
        format!("report_interval_ticks could not be converted to u32: {}", config.report_interval_ticks)
    })?;
    let max_ticks = if config.max_ticks > 0 {
        Some(
            u32::try_from(config.max_ticks)
                .with_context(|| format!("max_ticks could not be converted to u32: {}", config.max_ticks))?,
        )
    } else {
        None
    };

    let mut logger = Logger::new(run_dir, config)?;
    let mut state = SimulationState::init_from_config(config)?;
    let mut last_logged_tick = 0_u32;

    loop {
        if signal::is_signal_pending() {
            break;
        }

        let stats = state.tick()?;
        if stats.tick % report_interval == 0 {
            last_logged_tick = log_report_snapshot(&mut state, &mut logger)?;
        }
        if let Some(limit) = max_ticks
            && stats.tick >= limit
        {
            break;
        }
    }

    let final_tick = state.ui_stats()?.tick;
    if final_tick > 0 && final_tick != last_logged_tick {
        last_logged_tick = log_report_snapshot(&mut state, &mut logger)?;
    }

    logger.flush()?;
    stdout_line(&format!("Completed headless run: {run_label}"));
    stdout_line(&format!("Output directory: {}", logger.run_dir().display()));
    stdout_line(&format!("Final logged tick: {last_logged_tick}"));
    Ok(())
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
        if args.name.is_some() {
            stderr_line("Error: --name cannot be used with --all");
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
            let run_dir = resolve_run_dir(&config.output_dir, Some(name.as_str()), config.seed);
            run_headless_experiment(name, &config, &run_dir)?;
            if signal::is_signal_pending() {
                break;
            }
        }
        stdout_line("All experiments completed.");
        return Ok(());
    }

    let (selected_name, config) = select_named_experiment(&experiments, args.experiment.as_deref())?;
    let mut config = config;
    if let Some(ticks) = args.ticks {
        config.max_ticks = ticks;
    }

    if args.headless {
        let run_name = resolve_run_name(&selected_name, args.name.as_deref());
        let run_dir = resolve_run_dir(&config.output_dir, Some(run_name.as_str()), config.seed);
        return run_headless_experiment(&selected_name, &config, &run_dir);
    }

    validate_config(&config)?;

    stdout_line("MoonAI - GPU-first predator-prey evolution simulation");
    stdout_line(&format!("Loaded experiment: {selected_name}"));
    stdout_line(&format!("Config: {config:?}"));

    Ok(())
}
