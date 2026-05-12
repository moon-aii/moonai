use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use eframe::egui::{self, Button, DragValue, RichText};

use crate::experiment::{Experiment, ExperimentCatalog, SimulationConfig, validate_config};
use crate::settings::{AppSettings, UiConfig, save_settings};
use crate::ui::run_queue::{QueuedRun, RunOutcome, RunQueue, RunRecord};
use crate::ui::session::{RunSession, apply_text_style};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AppView {
    Experiments,
    Queue,
    Run,
    Settings,
}

pub struct App {
    root_dir: PathBuf,
    experiments: ExperimentCatalog,
    experiment_search: String,
    selected_experiment_index: usize,
    draft_config: SimulationConfig,
    settings: AppSettings,
    settings_dirty: bool,
    active_view: AppView,
    queue: RunQueue,
    active_session: Option<RunSession>,
    status_message: Option<String>,
    error_message: Option<String>,
}

impl App {
    pub fn run(root_dir: &Path, experiments: ExperimentCatalog, settings: AppSettings) -> Result<()> {
        let native_options = eframe::NativeOptions {
            renderer: eframe::Renderer::Wgpu,
            viewport: egui::ViewportBuilder::default()
                .with_title("MoonAI")
                .with_inner_size([settings.ui.window_width as f32, settings.ui.window_height as f32]),
            ..Default::default()
        };

        let root_dir = root_dir.to_path_buf();
        eframe::run_native(
            "MoonAI",
            native_options,
            Box::new(move |creation_context| {
                Self::new(creation_context, root_dir.clone(), experiments.clone(), settings)
                    .map(|app| -> Box<dyn eframe::App> { Box::new(app) })
                    .map_err(|error| -> Box<dyn std::error::Error + Send + Sync> {
                        Box::new(std::io::Error::other(error.to_string()))
                    })
            }),
        )
        .map_err(|error| anyhow::anyhow!(error.to_string()))
    }

    fn new(
        creation_context: &eframe::CreationContext<'_>,
        root_dir: PathBuf,
        experiments: ExperimentCatalog,
        settings: AppSettings,
    ) -> Result<Self> {
        apply_text_style(&creation_context.egui_ctx, &settings.ui);
        let selected_experiment_index = experiments.default_index();
        let draft_config = experiments
            .get(selected_experiment_index)
            .map_or_else(|| experiments.default().simulation_config, |experiment| experiment.simulation_config);
        Ok(Self {
            root_dir,
            experiments,
            experiment_search: String::new(),
            selected_experiment_index,
            draft_config,
            settings,
            settings_dirty: false,
            active_view: AppView::Experiments,
            queue: RunQueue::default(),
            active_session: None,
            status_message: None,
            error_message: None,
        })
    }

    fn current_experiment(&self) -> &Experiment {
        self.experiments.get(self.selected_experiment_index).unwrap_or_else(|| self.experiments.default())
    }

    fn select_experiment(&mut self, index: usize) {
        if self.selected_experiment_index == index {
            return;
        }
        if let Some(experiment) = self.experiments.get(index) {
            self.selected_experiment_index = index;
            self.draft_config = experiment.simulation_config;
            self.status_message = Some(format!("Loaded experiment '{}' into the editor", experiment.name));
        }
    }

    fn draft_error(&self) -> Option<String> {
        validate_config(&self.draft_config).err().map(|error| error.to_string())
    }

    fn make_run_request(&mut self) -> QueuedRun {
        QueuedRun {
            id: self.queue.reserve_id(),
            experiment_name: self.current_experiment().name.clone(),
            simulation_config: self.draft_config,
        }
    }

    fn start_run_now(&mut self, frame: &mut eframe::Frame) -> Result<()> {
        if let Some(error) = self.draft_error() {
            self.error_message = Some(error);
            return Ok(());
        }

        let run = self.make_run_request();
        if self.active_session.is_some() {
            self.finish_active_session(RunOutcome::Replaced)?;
        }
        self.start_session(frame, run)?;
        self.active_view = AppView::Run;
        Ok(())
    }

    fn enqueue_current_draft(&mut self, frame: &mut eframe::Frame) -> Result<()> {
        if let Some(error) = self.draft_error() {
            self.error_message = Some(error);
            return Ok(());
        }

        let experiment_name = self.current_experiment().name.clone();
        let queued_id = self.queue.enqueue(experiment_name.clone(), self.draft_config);
        self.status_message = Some(format!("Queued '{experiment_name}' as item #{queued_id}"));
        if self.active_session.is_none() {
            self.start_next_queued_run(frame)?;
        }
        Ok(())
    }

    fn start_session(&mut self, frame: &mut eframe::Frame, queued_run: QueuedRun) -> Result<()> {
        let render_state = frame
            .wgpu_render_state()
            .context("eframe did not provide a wgpu render state for the GPU world renderer")?;
        let session = RunSession::new(render_state, queued_run, self.settings.ui.clone())?;
        self.status_message = Some(format!("Started run '{}'", session.run_name()));
        self.active_session = Some(session);
        Ok(())
    }

    fn start_next_queued_run(&mut self, frame: &mut eframe::Frame) -> Result<()> {
        while self.active_session.is_none() {
            let Some(queued_run) = self.queue.pop_next() else {
                break;
            };
            let queued_id = queued_run.id;
            let experiment_name = queued_run.experiment_name.clone();
            match self.start_session(frame, queued_run) {
                Ok(()) => return Ok(()),
                Err(error) => {
                    let message = error.to_string();
                    self.queue.record(RunRecord {
                        id: queued_id,
                        experiment_name,
                        run_name: "failed_to_start".to_owned(),
                        output_dir: PathBuf::new(),
                        final_tick: 0,
                        outcome: RunOutcome::Failed(message.clone()),
                    });
                    self.error_message = Some(message);
                }
            }
        }
        Ok(())
    }

    fn finish_active_session(&mut self, outcome: RunOutcome) -> Result<()> {
        let Some(session) = self.active_session.take() else {
            return Ok(());
        };
        let record = session.finalize(outcome)?;
        self.status_message =
            Some(format!("Run '{}' {} at tick {}", record.experiment_name, record.outcome.label(), record.final_tick));
        self.queue.record(record);
        Ok(())
    }

    fn stop_active_session(&mut self, frame: &mut eframe::Frame) -> Result<()> {
        self.finish_active_session(RunOutcome::Stopped)?;
        self.start_next_queued_run(frame)
    }

    fn drive_active_session(&mut self, frame: &mut eframe::Frame) -> Result<()> {
        if let Some(session) = &mut self.active_session
            && let Err(error) = session.step()
        {
            let message = error.to_string();
            self.finish_active_session(RunOutcome::Failed(message.clone()))?;
            self.error_message = Some(message);
        }
        if self.active_session.as_ref().is_some_and(RunSession::is_finished) {
            self.finish_active_session(RunOutcome::Completed)?;
        }
        if self.active_session.is_none() {
            self.start_next_queued_run(frame)?;
        }
        Ok(())
    }

    fn apply_settings(&mut self, frame: &mut eframe::Frame, ctx: &egui::Context) -> Result<()> {
        apply_text_style(ctx, &self.settings.ui);
        ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(
            self.settings.ui.window_width as f32,
            self.settings.ui.window_height as f32,
        )));
        if let Some(session) = &mut self.active_session {
            let render_state = frame
                .wgpu_render_state()
                .context("eframe did not provide a wgpu render state for the GPU world renderer")?;
            session.apply_ui_config(render_state, self.settings.ui.clone())?;
        }
        Ok(())
    }

    fn draw_top_bar(&mut self, ui: &mut egui::Ui) {
        egui::Panel::top("moonai_app_top_bar").show_inside(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                for (view, label) in [
                    (AppView::Experiments, "Experiments"),
                    (AppView::Queue, "Queue"),
                    (AppView::Run, "Run"),
                    (AppView::Settings, "Settings"),
                ] {
                    ui.selectable_value(&mut self.active_view, view, label);
                }

                ui.separator();
                ui.label(format!("Loaded: {}", self.experiments.len()));
                ui.label(format!("Queued: {}", self.queue.pending().len()));
                if let Some(session) = &self.active_session {
                    ui.label(format!("Active: {} @ tick {}", session.experiment_name(), session.ui_stats().tick));
                } else {
                    ui.label("Active: idle");
                }

                if self.settings_dirty {
                    ui.separator();
                    ui.colored_label(egui::Color32::YELLOW, "Settings unsaved");
                }
            });

            if let Some(message) = &self.status_message {
                ui.colored_label(egui::Color32::LIGHT_GREEN, message);
            }
            if let Some(error) = &self.error_message {
                ui.colored_label(egui::Color32::LIGHT_RED, error);
            }
        });
    }

    fn draw_run_view(&mut self, ui: &mut egui::Ui) {
        if let Some(session) = &mut self.active_session {
            session.render(ui);
        } else {
            egui::CentralPanel::default().show_inside(ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.heading("No active run");
                    if self.queue.is_empty() {
                        ui.label("The queue is empty. Start a run or add one to the queue from the Experiments tab.");
                    } else {
                        ui.label("The queue will start automatically on the next frame.");
                    }
                });
            });
        }
    }

    fn draw_experiments_view(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) -> Result<()> {
        egui::Panel::left("moonai_experiment_list").resizable(false).default_size(280.0).show_inside(ui, |ui| {
            ui.heading("Experiments");
            ui.label(format!("Source: {}", self.experiments.path().display()));
            ui.add(
                egui::TextEdit::singleline(&mut self.experiment_search)
                    .hint_text("Search experiments")
                    .desired_width(f32::INFINITY),
            );
            ui.separator();
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                let items: Vec<(usize, String, bool)> = self
                    .experiments
                    .experiments()
                    .iter()
                    .enumerate()
                    .map(|(index, experiment)| (index, experiment.name.clone(), experiment.is_default))
                    .filter(|(_, name, _)| {
                        self.experiment_search.is_empty()
                            || name.to_ascii_lowercase().contains(&self.experiment_search.to_ascii_lowercase())
                    })
                    .collect();
                ui.label(format!("Showing {} of {}", items.len(), self.experiments.len()));
                ui.separator();
                for (index, name, is_default) in items {
                    let selected = self.selected_experiment_index == index;
                    let label = if is_default { format!("{name} (default)") } else { name };
                    if ui.selectable_label(selected, label).clicked() {
                        self.select_experiment(index);
                    }
                }
            });
        });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                let experiment_name = self.current_experiment().name.clone();
                let experiment_path = self.current_experiment().path.clone();
                let preset_config = self.current_experiment().simulation_config;
                let draft_valid = self.draft_error().is_none();

                ui.heading(&experiment_name);
                ui.label(format!("Preset file: {}", experiment_path.display()));
                if preset_config != self.draft_config {
                    ui.label(RichText::new("Draft differs from preset").italics());
                }
                ui.horizontal_wrapped(|ui| {
                    summary_chip(ui, "Seed", self.draft_config.seed.to_string());
                    summary_chip(ui, "Ticks", self.draft_config.max_ticks.to_string());
                    summary_chip(
                        ui,
                        "Population",
                        (self.draft_config.predator_count + self.draft_config.prey_count).to_string(),
                    );
                    summary_chip(ui, "Food", self.draft_config.food_count.to_string());
                    summary_chip(ui, "Report", self.draft_config.report_interval_ticks.to_string());
                });
                if self.draft_config.max_ticks == 0 {
                    ui.colored_label(
                        egui::Color32::YELLOW,
                        "max_ticks is 0, so this run will not finish on its own and will block queued runs until it is stopped or replaced.",
                    );
                }
                if let Some(error) = self.draft_error() {
                    ui.colored_label(egui::Color32::LIGHT_RED, error);
                } else {
                    ui.colored_label(egui::Color32::LIGHT_GREEN, "Draft is valid.");
                }

                section_card(ui, "Run Actions", |ui| {
                    ui.horizontal_wrapped(|ui| {
                        let run_label = if self.active_session.is_some() { "Replace Current Run" } else { "Start Run" };
                        if ui.add_enabled(draft_valid, Button::new(run_label)).clicked()
                            && let Err(error) = self.start_run_now(frame)
                        {
                            self.error_message = Some(error.to_string());
                        }
                        if ui.add_enabled(draft_valid, Button::new("Add To Queue")).clicked()
                            && let Err(error) = self.enqueue_current_draft(frame)
                        {
                            self.error_message = Some(error.to_string());
                        }
                        if ui.button("Reset Draft From Preset").clicked() {
                            self.draft_config = preset_config;
                            self.status_message = Some(format!("Reset draft to '{experiment_name}'."));
                        }
                        if ui.button("Open Queue").clicked() {
                            self.active_view = AppView::Queue;
                        }
                    });
                });

                section_card(ui, "Preset Summary", |ui| {
                    ui.horizontal_wrapped(|ui| {
                        summary_chip(ui, "Grid", format!("{:.0}", self.draft_config.grid_size));
                        summary_chip(ui, "Predators", self.draft_config.predator_count.to_string());
                        summary_chip(ui, "Prey", self.draft_config.prey_count.to_string());
                        summary_chip(ui, "Vision", format!("{:.2}", self.draft_config.vision_range));
                        summary_chip(ui, "Respawn", format!("{:.3}", self.draft_config.food_respawn_rate));
                        summary_chip(
                            ui,
                            "Species Threshold",
                            format!("{:.2}", self.draft_config.compatibility_threshold),
                        );
                    });
                });

                ui.separator();
                draw_simulation_config_editor(ui, &mut self.draft_config);
            });
        });
        Ok(())
    }

    fn draw_queue_view(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) -> Result<()> {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            let mut stop_requested = false;
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                ui.heading("Run Queue");

                if let Some(session) = &self.active_session {
                    section_card(ui, "Current Run", |ui| {
                        ui.label(format!("Experiment: {}", session.experiment_name()));
                        ui.label(format!("Run: {}", session.run_name()));
                        ui.label(format!("Tick: {}", session.ui_stats().tick));
                        ui.label(format!("Output: {}", session.output_dir().display()));
                        ui.horizontal_wrapped(|ui| {
                            if ui.button("Open Run Tab").clicked() {
                                self.active_view = AppView::Run;
                            }
                            if ui.button("Stop Current Run").clicked() {
                                stop_requested = true;
                            }
                        });
                    });
                } else {
                    section_card(ui, "Current Run", |ui| {
                        ui.label("No active run.");
                        if !self.queue.is_empty() {
                            ui.label("The next queued run will start automatically.");
                        }
                    });
                }

                ui.separator();
                let pending_runs: Vec<_> = self.queue.pending().iter().cloned().collect();
                section_card(ui, "Pending", |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(format!("{} pending run(s)", pending_runs.len()));
                        if ui.button("Open Experiments").clicked() {
                            self.active_view = AppView::Experiments;
                        }
                    });
                    if pending_runs.is_empty() {
                        ui.label("Queue is empty.");
                        return;
                    }
                    let mut remove_id = None;
                    for run in pending_runs {
                        ui.group(|ui| {
                            ui.horizontal_wrapped(|ui| {
                                ui.strong(format!("#{} {}", run.id, run.experiment_name));
                                summary_chip(ui, "Ticks", run.simulation_config.max_ticks.to_string());
                                summary_chip(ui, "Seed", run.simulation_config.seed.to_string());
                                summary_chip(
                                    ui,
                                    "Population",
                                    (run.simulation_config.predator_count + run.simulation_config.prey_count)
                                        .to_string(),
                                );
                            });
                            if ui.button("Remove").clicked() {
                                remove_id = Some(run.id);
                            }
                        });
                    }
                    if let Some(id) = remove_id {
                        let _ = self.queue.remove_pending(id);
                        self.status_message = Some(format!("Removed queue item #{id}"));
                    }
                });

                ui.separator();
                section_card(ui, "History", |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(format!("{} recorded run(s)", self.queue.history().len()));
                        if ui.button("Clear History").clicked() {
                            self.queue.clear_history();
                        }
                    });
                    if self.queue.history().is_empty() {
                        ui.label("No completed or failed runs yet.");
                        return;
                    }
                    for record in self.queue.history() {
                        ui.group(|ui| {
                            ui.horizontal_wrapped(|ui| {
                                ui.strong(format!("#{} {}", record.id, record.experiment_name));
                                summary_chip(ui, "Status", record.outcome.label().to_owned());
                                summary_chip(ui, "Final Tick", record.final_tick.to_string());
                            });
                            ui.label(format!("Output: {}", record.output_dir.display()));
                        });
                        if let RunOutcome::Failed(message) = &record.outcome {
                            ui.colored_label(egui::Color32::LIGHT_RED, message);
                        }
                    }
                });
            });

            if stop_requested && let Err(error) = self.stop_active_session(frame) {
                self.error_message = Some(error.to_string());
            }
        });
        Ok(())
    }

    fn draw_settings_view(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) -> Result<()> {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            let mut changed = false;
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                ui.heading("Settings");
                ui.label(format!("File: {}", self.root_dir.join("settings.json").display()));
                ui.label("Changes apply live. Save persists them for the next launch.");
                if self.settings_dirty {
                    ui.colored_label(egui::Color32::YELLOW, "Unsaved changes");
                }
                section_card(ui, "Actions", |ui| {
                    ui.horizontal_wrapped(|ui| {
                        if ui.button("Save Settings").clicked() {
                            match save_settings(&self.root_dir, &self.settings) {
                                Ok(()) => {
                                    self.settings_dirty = false;
                                    self.status_message = Some("Saved settings.json".to_owned());
                                }
                                Err(error) => self.error_message = Some(error.to_string()),
                            }
                        }
                        if ui.button("Reset To Defaults").clicked() {
                            self.settings.ui = UiConfig::default();
                            self.settings_dirty = true;
                            changed = true;
                        }
                    });
                });

                ui.separator();
                changed |= draw_ui_config_editor(ui, &mut self.settings.ui);
            });

            if changed {
                self.settings_dirty = true;
                if let Err(error) = self.apply_settings(frame, ui.ctx()) {
                    self.error_message = Some(error.to_string());
                }
            }
        });
        Ok(())
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        self.error_message = None;
        if let Some(session) = &mut self.active_session
            && self.active_view == AppView::Run
        {
            session.prepare_visible_frame(ui.ctx(), frame);
        }
        if let Err(error) = self.drive_active_session(frame) {
            self.error_message = Some(error.to_string());
        }

        self.draw_top_bar(ui);
        match self.active_view {
            AppView::Experiments => {
                if let Err(error) = self.draw_experiments_view(ui, frame) {
                    self.error_message = Some(error.to_string());
                }
            }
            AppView::Queue => {
                if let Err(error) = self.draw_queue_view(ui, frame) {
                    self.error_message = Some(error.to_string());
                }
            }
            AppView::Run => self.draw_run_view(ui),
            AppView::Settings => {
                if let Err(error) = self.draw_settings_view(ui, frame) {
                    self.error_message = Some(error.to_string());
                }
            }
        }

        if let Some(session) = &self.active_session {
            session.request_repaint(ui.ctx());
        }
    }

    fn on_exit(&mut self) {
        let _ = self.finish_active_session(RunOutcome::Stopped);
    }
}

fn draw_simulation_config_editor(ui: &mut egui::Ui, config: &mut SimulationConfig) {
    grouped_section(ui, "Population", |ui| {
        f32_row(ui, "grid_size", &mut config.grid_size, 1.0);
        u32_row(ui, "predator_count", &mut config.predator_count, 1.0);
        u32_row(ui, "prey_count", &mut config.prey_count, 1.0);
        u32_row(ui, "food_count", &mut config.food_count, 1.0);
        u32_row(ui, "max_age", &mut config.max_age, 1.0);
        u32_row(ui, "max_ticks", &mut config.max_ticks, 100.0);
        u32_row(ui, "report_interval_ticks", &mut config.report_interval_ticks, 10.0);
        u64_row(ui, "seed", &mut config.seed, 1.0);
    });
    grouped_section(ui, "Movement And Sensing", |ui| {
        f32_row(ui, "predator_speed", &mut config.predator_speed, 0.01);
        f32_row(ui, "prey_speed", &mut config.prey_speed, 0.01);
        f32_row(ui, "vision_range", &mut config.vision_range, 0.1);
        f32_row(ui, "interaction_range", &mut config.interaction_range, 0.1);
        f32_row(ui, "mate_range", &mut config.mate_range, 0.1);
    });
    grouped_section(ui, "Energy And Lifecycle", |ui| {
        f32_row(ui, "food_respawn_rate", &mut config.food_respawn_rate, 0.001);
        f32_row(ui, "energy_drain_per_tick", &mut config.energy_drain_per_tick, 0.001);
        f32_row(ui, "energy_gain_from_kill", &mut config.energy_gain_from_kill, 0.01);
        f32_row(ui, "energy_gain_from_food", &mut config.energy_gain_from_food, 0.01);
        f32_row(ui, "initial_energy", &mut config.initial_energy, 0.01);
        f32_row(ui, "max_energy", &mut config.max_energy, 0.01);
        f32_row(ui, "reproduction_energy_threshold", &mut config.reproduction_energy_threshold, 0.01);
        f32_row(ui, "reproduction_energy_cost", &mut config.reproduction_energy_cost, 0.01);
        f32_row(ui, "offspring_initial_energy", &mut config.offspring_initial_energy, 0.01);
    });
    grouped_section(ui, "Evolution And Speciation", |ui| {
        f32_row(ui, "mutation_rate", &mut config.mutation_rate, 0.001);
        f32_row(ui, "weight_mutation_power", &mut config.weight_mutation_power, 0.001);
        f32_row(ui, "add_node_rate", &mut config.add_node_rate, 0.001);
        f32_row(ui, "add_connection_rate", &mut config.add_connection_rate, 0.001);
        f32_row(ui, "delete_connection_rate", &mut config.delete_connection_rate, 0.000001);
        u32_row(ui, "max_hidden_nodes", &mut config.max_hidden_nodes, 1.0);
        f32_row(ui, "compatibility_threshold", &mut config.compatibility_threshold, 0.1);
        f32_row(ui, "compatibility_min_normalization", &mut config.compatibility_min_normalization, 0.1);
        f32_row(ui, "c1_excess", &mut config.c1_excess, 0.01);
        f32_row(ui, "c2_disjoint", &mut config.c2_disjoint, 0.01);
        f32_row(ui, "c3_weight", &mut config.c3_weight, 0.01);
    });
}

fn draw_ui_config_editor(ui: &mut egui::Ui, config: &mut UiConfig) -> bool {
    let mut changed = false;
    grouped_section(ui, "Window And Layout", |ui| {
        changed |= u32_row(ui, "window_width", &mut config.window_width, 1.0);
        changed |= u32_row(ui, "window_height", &mut config.window_height, 1.0);
        changed |= f32_row(ui, "left_panel_width", &mut config.left_panel_width, 1.0);
        changed |= f32_row(ui, "right_panel_width", &mut config.right_panel_width, 1.0);
        changed |= f32_row(ui, "font_size", &mut config.font_size, 0.5);
        changed |= f32_row(ui, "simulation_margin", &mut config.simulation_margin, 0.5);
        changed |= u32_row(ui, "fps_limit", &mut config.fps_limit, 1.0);
        changed |= f32_row(ui, "nn_panel_margin", &mut config.nn_panel_margin, 0.5);
        changed |= f32_row(ui, "nn_panel_gap", &mut config.nn_panel_gap, 0.5);
        changed |= f32_row(ui, "selected_info_panel_height", &mut config.selected_info_panel_height, 0.5);
        changed |= f32_row(ui, "nn_panel_top_reserve", &mut config.nn_panel_top_reserve, 0.5);
        changed |= f32_row(ui, "nn_panel_min_height", &mut config.nn_panel_min_height, 0.5);
        changed |= f32_row(ui, "nn_panel_min_width", &mut config.nn_panel_min_width, 0.5);
        changed |= f32_row(ui, "nn_panel_max_width", &mut config.nn_panel_max_width, 0.5);
        changed |= f32_row(ui, "selection_click_radius", &mut config.selection_click_radius, 0.5);
        changed |= f32_row(ui, "fps_alpha", &mut config.fps_alpha, 0.01);
        changed |= f32_row(ui, "zoom_min", &mut config.zoom_min, 0.01);
        changed |= f32_row(ui, "zoom_max", &mut config.zoom_max, 0.01);
        changed |= u32_row(ui, "speed_min", &mut config.speed_min, 1.0);
        changed |= u32_row(ui, "speed_max", &mut config.speed_max, 1.0);
    });
    grouped_section(ui, "Entity Rendering", |ui| {
        changed |= f32_row(ui, "predator_size", &mut config.predator_size, 0.1);
        changed |= f32_row(ui, "prey_size", &mut config.prey_size, 0.1);
        changed |= f32_row(ui, "food_size", &mut config.food_size, 0.1);
        changed |= f32_row(ui, "triangle_tip_factor", &mut config.triangle_tip_factor, 0.05);
        changed |= f32_row(ui, "triangle_base_factor", &mut config.triangle_base_factor, 0.05);
        changed |= f32_row(ui, "triangle_width_factor", &mut config.triangle_width_factor, 0.05);
        changed |= u32_row(ui, "circle_point_count", &mut config.circle_point_count, 1.0);
        changed |= u32_row(ui, "vision_point_count", &mut config.vision_point_count, 1.0);
        changed |= f32_row(ui, "selected_outline_thickness", &mut config.selected_outline_thickness, 0.1);
    });
    grouped_section(ui, "World Colors", |ui| {
        changed |= rgb_row(ui, "predator_color", &mut config.predator_color);
        changed |= rgb_row(ui, "prey_color", &mut config.prey_color);
        changed |= rgb_row(ui, "food_color", &mut config.food_color);
        changed |= rgb_row(ui, "grid_color", &mut config.grid_color);
        changed |= rgb_row(ui, "border_color", &mut config.border_color);
        changed |= rgb_row(ui, "bg_color", &mut config.bg_color);
    });
    grouped_section(ui, "Panels And Overlay", |ui| {
        changed |= rgb_row(ui, "panel_bg_color", &mut config.panel_bg_color);
        changed |= f32_row(ui, "panel_alpha", &mut config.panel_alpha, 1.0);
        changed |= rgb_row(ui, "panel_outline_color", &mut config.panel_outline_color);
        changed |= rgba_row(ui, "vision_fill_color", &mut config.vision_fill_color);
        changed |= f32_row(ui, "vision_fill_alpha", &mut config.vision_fill_alpha, 1.0);
        changed |= f32_row(ui, "vision_outline_alpha", &mut config.vision_outline_alpha, 1.0);
        changed |= f32_row(ui, "sensor_alpha", &mut config.sensor_alpha, 1.0);
        changed |= f32_row(ui, "food_sensor_alpha", &mut config.food_sensor_alpha, 1.0);
        changed |= f32_row(ui, "food_alpha", &mut config.food_alpha, 1.0);
        changed |= f32_row(ui, "bar_alpha", &mut config.bar_alpha, 1.0);
    });
    grouped_section(ui, "Text And Charts", |ui| {
        changed |= rgb_row(ui, "title_color", &mut config.title_color);
        changed |= rgb_row(ui, "fitness_color", &mut config.fitness_color);
        changed |= rgb_row(ui, "muted_color", &mut config.muted_color);
        changed |= rgb_row(ui, "pause_color", &mut config.pause_color);
        changed |= rgb_row(ui, "event_kill_color", &mut config.event_kill_color);
        changed |= rgb_row(ui, "event_food_color", &mut config.event_food_color);
        changed |= rgb_row(ui, "event_birth_color", &mut config.event_birth_color);
        changed |= rgb_row(ui, "event_death_color", &mut config.event_death_color);
        changed |= rgb_row(ui, "chart_best_color", &mut config.chart_best_color);
        changed |= rgb_row(ui, "chart_avg_color", &mut config.chart_avg_color);
    });
    grouped_section(ui, "Network And Energy Colors", |ui| {
        changed |= rgba_row(ui, "nn_node_outline_color", &mut config.nn_node_outline_color);
        changed |= rgb_row(ui, "nn_input_color", &mut config.nn_input_color);
        changed |= rgb_row(ui, "nn_bias_color", &mut config.nn_bias_color);
        changed |= rgb_row(ui, "nn_hidden_color", &mut config.nn_hidden_color);
        changed |= rgb_row(ui, "nn_output_color", &mut config.nn_output_color);
        changed |= rgb_row(ui, "energy_bucket_0", &mut config.energy_bucket_0);
        changed |= rgb_row(ui, "energy_bucket_1", &mut config.energy_bucket_1);
        changed |= rgb_row(ui, "energy_bucket_2", &mut config.energy_bucket_2);
        changed |= rgb_row(ui, "energy_bucket_3", &mut config.energy_bucket_3);
        changed |= rgb_row(ui, "energy_bucket_4", &mut config.energy_bucket_4);
    });
    changed
}

fn grouped_section(ui: &mut egui::Ui, title: &str, add_contents: impl FnOnce(&mut egui::Ui)) {
    egui::CollapsingHeader::new(title).default_open(true).show(ui, |ui| {
        add_contents(ui);
    });
}

fn section_card(ui: &mut egui::Ui, title: &str, add_contents: impl FnOnce(&mut egui::Ui)) {
    ui.group(|ui| {
        ui.heading(title);
        add_contents(ui);
    });
}

fn summary_chip(ui: &mut egui::Ui, label: &str, value: String) {
    ui.group(|ui| {
        ui.horizontal(|ui| {
            ui.label(RichText::new(label).small().weak());
            ui.strong(value);
        });
    });
}

fn f32_row(ui: &mut egui::Ui, label: &str, value: &mut f32, speed: f64) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        changed = ui.add(DragValue::new(value).speed(speed)).changed();
    });
    changed
}

fn u32_row(ui: &mut egui::Ui, label: &str, value: &mut u32, speed: f64) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        changed = ui.add(DragValue::new(value).speed(speed)).changed();
    });
    changed
}

fn u64_row(ui: &mut egui::Ui, label: &str, value: &mut u64, speed: f64) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        changed = ui.add(DragValue::new(value).speed(speed)).changed();
    });
    changed
}

fn rgb_row(ui: &mut egui::Ui, label: &str, value: &mut [f32; 3]) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        changed = ui.color_edit_button_rgb(value).changed();
    });
    changed
}

fn rgba_row(ui: &mut egui::Ui, label: &str, value: &mut [f32; 4]) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        changed = ui.color_edit_button_rgba_unmultiplied(value).changed();
    });
    changed
}
