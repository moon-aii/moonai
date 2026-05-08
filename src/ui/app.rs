use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result};
use eframe::egui::{self, Key, Sense, TextureHandle, TextureOptions};

use crate::config::SimulationConfig;
use crate::settings::UiConfig;
use crate::tick::buffers::{RenderAgentReadback, RenderSnapshotReadback, UiStatsReadback};
use crate::tick::genome::PopulationKind;
use crate::tick::simulation::SimulationState;
use crate::ui::render;
use crate::ui::types::{CameraState, OverlayStats, SelectedAgent, SelectedAgentData, UiState};

pub struct App {
    run_label: String,
    config: SimulationConfig,
    ui_config: UiConfig,
    state: SimulationState,
    ui_state: UiState,
    camera: CameraState,
    ui_stats: UiStatsReadback,
    snapshot: RenderSnapshotReadback,
    selected: Option<SelectedAgent>,
    selected_data: Option<SelectedAgentData>,
    texture: Option<TextureHandle>,
    last_frame_started: Instant,
    fps: f32,
    latest_scene: Option<egui::ColorImage>,
    status_message: Option<String>,
    error_message: Option<String>,
    reached_tick_limit: bool,
}

impl App {
    pub fn run(run_label: &str, config: &SimulationConfig, ui_config: &UiConfig) -> Result<()> {
        let icon = load_icon();
        let mut native_options = eframe::NativeOptions {
            renderer: eframe::Renderer::Wgpu,
            viewport: egui::ViewportBuilder::default()
                .with_title(format!("MoonAI - {run_label}"))
                .with_inner_size([ui_config.window_width as f32, ui_config.window_height as f32]),
            ..Default::default()
        };
        if let Some(icon) = icon {
            native_options.viewport = native_options.viewport.with_icon(icon);
        }

        let run_label = run_label.to_owned();
        let config_for_app = config.clone();
        let ui_for_app = ui_config.clone();
        eframe::run_native(
            "MoonAI",
            native_options,
            Box::new(move |_creation_context| {
                Self::new(&run_label, config_for_app.clone(), ui_for_app.clone())
                    .map(|app| -> Box<dyn eframe::App> { Box::new(app) })
                    .map_err(|error| -> Box<dyn std::error::Error + Send + Sync> {
                        Box::new(std::io::Error::other(error.to_string()))
                    })
            }),
        )
        .map_err(|error| anyhow::anyhow!(error.to_string()))
    }

    fn new(run_label: &str, config: SimulationConfig, ui_config: UiConfig) -> Result<Self> {
        let mut state = SimulationState::init_from_config(&config)?;
        let camera = render::default_camera(config.grid_size as f32);
        let (ui_stats, snapshot) = refresh_snapshot(&mut state)?;

        Ok(Self {
            run_label: run_label.to_owned(),
            config,
            ui_config,
            state,
            ui_state: UiState::default(),
            camera,
            ui_stats,
            snapshot,
            selected: None,
            selected_data: None,
            texture: None,
            last_frame_started: Instant::now(),
            fps: 0.0,
            latest_scene: None,
            status_message: None,
            error_message: None,
            reached_tick_limit: false,
        })
    }

    fn tick_limit(&self) -> Option<u32> {
        (self.config.max_ticks > 0).then_some(self.config.max_ticks as u32)
    }

    fn handle_shortcuts(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        if ctx.input(|input| input.key_pressed(Key::Space)) {
            self.ui_state.paused = !self.ui_state.paused;
            self.status_message =
                Some(if self.ui_state.paused { "Simulation paused" } else { "Simulation resumed" }.to_owned());
        }
        if ctx.input(|input| input.key_pressed(Key::ArrowUp)) {
            self.increase_speed();
        }
        if ctx.input(|input| input.key_pressed(Key::ArrowDown)) {
            self.decrease_speed();
        }
        if ctx.input(|input| input.key_pressed(Key::Home)) {
            self.camera = render::default_camera(self.config.grid_size as f32);
        }
        if ctx.input(|input| input.key_pressed(Key::Escape)) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        if ctx.input(|input| input.key_pressed(Key::S))
            && let Err(error) = self.save_screenshot()
        {
            self.error_message = Some(error.to_string());
        }

        let mut request_step = false;
        let mut request_speed_up = false;
        let mut request_speed_down = false;
        ctx.input(|input| {
            for event in &input.events {
                if let egui::Event::Text(text) = event {
                    match text.as_str() {
                        "." => request_step = true,
                        "+" | "=" => request_speed_up = true,
                        "-" | "_" => request_speed_down = true,
                        _ => {}
                    }
                }
            }
        });
        if request_step && self.ui_state.paused {
            self.ui_state.tick_requested = true;
        }
        if request_speed_up {
            self.increase_speed();
        }
        if request_speed_down {
            self.decrease_speed();
        }

        let _ = frame;
    }

    fn increase_speed(&mut self) {
        let next = self.ui_state.speed_multiplier.saturating_mul(2);
        self.ui_state.speed_multiplier = next.clamp(self.ui_config.speed_min, self.ui_config.speed_max);
    }

    fn decrease_speed(&mut self) {
        self.ui_state.speed_multiplier = (self.ui_state.speed_multiplier / 2).max(self.ui_config.speed_min);
    }

    fn drive_simulation(&mut self) -> Result<()> {
        if self.reached_tick_limit {
            self.ui_state.paused = true;
            return Ok(());
        }

        let steps = if self.ui_state.paused {
            if self.ui_state.tick_requested { 1 } else { 0 }
        } else {
            self.ui_state.speed_multiplier
        };

        let limit = self.tick_limit();
        for _ in 0..steps {
            if let Some(limit) = limit
                && self.ui_stats.tick >= limit
            {
                self.reached_tick_limit = true;
                self.ui_state.paused = true;
                self.status_message = Some(format!("Reached tick limit at {limit}"));
                break;
            }

            self.ui_stats = self.state.tick()?;
        }
        self.ui_state.tick_requested = false;

        let (ui_stats, snapshot) = refresh_snapshot(&mut self.state)?;
        self.ui_stats = ui_stats;
        self.snapshot = snapshot;
        self.sync_selected_agent()?;
        Ok(())
    }

    fn sync_selected_agent(&mut self) -> Result<()> {
        let Some(selected) = self.selected else {
            self.selected_data = None;
            self.ui_state.selected_agent_id = None;
            return Ok(());
        };

        let Some(agent) = find_agent_by_entity(&self.snapshot, selected.population_kind, selected.entity_id) else {
            self.selected = None;
            self.selected_data = None;
            self.ui_state.selected_agent_id = None;
            return Ok(());
        };

        let resolved = SelectedAgent::from_render_agent(agent);
        self.selected = Some(resolved);
        self.ui_state.selected_agent_id = Some(resolved.entity_id);
        self.selected_data = Some(SelectedAgentData {
            agent: *agent,
            sensors: self.state.sensor_snapshot(agent.population_kind, agent.slot)?,
            network: self.state.selected_agent_network(agent.population_kind, agent.slot)?,
            genome: self.state.representative_genome(agent.population_kind, agent.slot)?,
        });
        Ok(())
    }

    fn save_screenshot(&mut self) -> Result<()> {
        let Some(image) = &self.latest_scene else {
            return Ok(());
        };

        let path = screenshot_path(&self.run_label, self.ui_stats.tick)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create screenshot directory {}", parent.display()))?;
        }
        render::save_png(&path, image)?;
        self.status_message = Some(format!("Saved screenshot to {}", path.display()));
        Ok(())
    }

    fn update_fps(&mut self) {
        let now = Instant::now();
        let delta = now.saturating_duration_since(self.last_frame_started);
        self.last_frame_started = now;
        let seconds = delta.as_secs_f32();
        if seconds > f32::EPSILON {
            let frame_fps = 1.0 / seconds;
            self.fps = if self.fps == 0.0 {
                frame_fps
            } else {
                (self.fps * (1.0 - self.ui_config.fps_alpha)) + (frame_fps * self.ui_config.fps_alpha)
            };
        }
    }

    fn draw_side_panel(&mut self, ui: &mut egui::Ui) {
        egui::Panel::right("moonai_side_panel")
            .resizable(false)
            .default_size(self.ui_config.ui_side_margin)
            .show_inside(ui, |ui| {
                ui.heading("MoonAI");
                ui.label(format!("Experiment: {}", self.run_label));

                let overlay = OverlayStats::from_snapshot(
                    &self.snapshot,
                    self.ui_stats,
                    self.ui_state.speed_multiplier,
                    self.ui_state.paused,
                    self.fps,
                );
                ui.separator();
                ui.label(format!("Tick: {}", overlay.ui_stats.tick));
                ui.label(format!("FPS: {:.1}", overlay.fps));
                ui.label(format!("Speed: {}x", overlay.speed_multiplier));
                ui.label(if overlay.paused { "State: paused" } else { "State: running" });
                ui.label(format!("Predators: {}", overlay.ui_stats.predator_count));
                ui.label(format!("Prey: {}", overlay.ui_stats.prey_count));
                ui.label(format!("Food rendered: {}", overlay.food_returned));
                ui.label(format!("Kills: {}", overlay.ui_stats.kills));
                ui.label(format!("Food eaten: {}", overlay.ui_stats.food_eaten));
                ui.label(format!(
                    "Predator births/deaths: {}/{}",
                    overlay.ui_stats.predator_births, overlay.ui_stats.predator_deaths
                ));
                ui.label(format!(
                    "Prey births/deaths: {}/{}",
                    overlay.ui_stats.prey_births, overlay.ui_stats.prey_deaths
                ));
                ui.label(format!("Avg predator energy: {:.3}", overlay.ui_stats.avg_predator_energy));
                ui.label(format!("Avg prey energy: {:.3}", overlay.ui_stats.avg_prey_energy));

                ui.separator();
                ui.label("Controls");
                ui.label("Space: pause/resume");
                ui.label("Up/Down or +/-: speed");
                ui.label(".: single tick while paused");
                ui.label("S: screenshot");
                ui.label("Home: reset camera");
                ui.label("Scroll: zoom");
                ui.label("Middle/right drag: pan");
                ui.label("Left click: select agent");

                if let Some(message) = &self.status_message {
                    ui.separator();
                    ui.colored_label(egui::Color32::LIGHT_GREEN, message);
                }
                if let Some(error) = &self.error_message {
                    ui.separator();
                    ui.colored_label(egui::Color32::LIGHT_RED, error);
                }

                if let Some(selected) = &self.selected_data {
                    ui.separator();
                    ui.heading("Selected Agent");
                    ui.label(format!("Population: {:?}", selected.agent.population_kind));
                    ui.label(format!("Entity: {}", selected.agent.entity_id));
                    ui.label(format!("Slot: {}", selected.agent.slot));
                    ui.label(format!("Species: {}", selected.agent.species_id));
                    ui.label(format!("Generation: {}", selected.agent.generation));
                    ui.label(format!("Energy: {:.3}", selected.agent.energy));
                    ui.label(format!("Outputs: {:.3}, {:.3}", selected.network.output_0, selected.network.output_1));
                    ui.label(format!("Nodes: {}", selected.network.node_count));
                    ui.label(format!("Connections: {}", selected.genome.header.num_connections));

                    ui.separator();
                    ui.label("Sensor Summary");
                    ui.label(format!("Energy input: {:.3}", selected.sensors.inputs[30]));
                    ui.label(format!(
                        "Velocity x/y: {:.3}, {:.3}",
                        selected.sensors.inputs[31], selected.sensors.inputs[32]
                    ));
                    ui.label(format!(
                        "Wall x/y: {:.3}, {:.3}",
                        selected.sensors.inputs[33], selected.sensors.inputs[34]
                    ));

                    ui.separator();
                    let width = ui
                        .available_width()
                        .clamp(self.ui_config.nn_panel_min_width, self.ui_config.nn_panel_max_width);
                    let height = self.ui_config.nn_panel_min_height.max(ui.available_height() - 8.0);
                    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), Sense::hover());
                    render::paint_network(ui, rect, &self.ui_config, selected);
                }
            });
    }

    fn draw_world(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            let available = ui.available_size();
            let width = available.x.max(1.0) as usize;
            let height = available.y.max(1.0) as usize;
            let image = render::render_scene(&render::SceneRenderInput {
                snapshot: &self.snapshot,
                ui_config: &self.ui_config,
                camera: self.camera,
                world_size: self.config.grid_size as f32,
                selected: self.selected_data.as_ref(),
                image_size: [width, height],
                vision_range: self.config.vision_range,
            });
            self.latest_scene = Some(image.clone());

            match &mut self.texture {
                Some(texture) => texture.set(image, TextureOptions::LINEAR),
                None => {
                    self.texture = Some(ui.ctx().load_texture("moonai_scene", image, TextureOptions::LINEAR));
                }
            }

            let Some(texture) = &self.texture else {
                return;
            };
            let response = ui.add(
                egui::Image::new((texture.id(), available.max(egui::vec2(1.0, 1.0)))).sense(Sense::click_and_drag()),
            );

            self.handle_view_input(ui.ctx(), response.rect, &response);
        });
    }

    fn handle_view_input(&mut self, ctx: &egui::Context, rect: egui::Rect, response: &egui::Response) {
        if response.hovered() {
            let scroll = ctx.input(|input| input.smooth_scroll_delta.y);
            if scroll.abs() > f32::EPSILON {
                let hover = ctx.input(|input| input.pointer.hover_pos()).unwrap_or(rect.center());
                let world_before = render::screen_to_world(rect, self.camera, self.config.grid_size as f32, hover);
                let zoom_factor = (scroll * 0.0015).exp();
                self.camera.zoom *= zoom_factor;
                render::clamp_camera(&mut self.camera, self.config.grid_size as f32, &self.ui_config);
                let world_after = render::screen_to_world(rect, self.camera, self.config.grid_size as f32, hover);
                self.camera.center_x += world_before.0 - world_after.0;
                self.camera.center_y += world_before.1 - world_after.1;
                render::clamp_camera(&mut self.camera, self.config.grid_size as f32, &self.ui_config);
            }
        }

        let dragging =
            response.dragged_by(egui::PointerButton::Middle) || response.dragged_by(egui::PointerButton::Secondary);
        if dragging {
            let delta = ctx.input(|input| input.pointer.delta());
            let scale =
                rect.width().min(rect.height()).max(1.0) / (self.config.grid_size as f32 / self.camera.zoom.max(0.001));
            self.camera.center_x -= delta.x / scale;
            self.camera.center_y += delta.y / scale;
            render::clamp_camera(&mut self.camera, self.config.grid_size as f32, &self.ui_config);
        }

        if response.clicked_by(egui::PointerButton::Primary)
            && let Some(pointer) = response.interact_pointer_pos()
        {
            self.select_agent(rect, pointer);
        }
    }

    fn select_agent(&mut self, rect: egui::Rect, pointer: egui::Pos2) {
        let mut closest: Option<(&RenderAgentReadback, f32)> = None;
        let select_radius = self.ui_config.selection_click_radius;

        for agent in self.snapshot.predators.iter().chain(self.snapshot.prey.iter()) {
            let screen =
                render::world_to_screen(rect, self.camera, self.config.grid_size as f32, agent.pos_x, agent.pos_y);
            let distance = screen.distance(pointer);
            if distance > select_radius {
                continue;
            }
            if closest.is_none_or(|(_, best)| distance < best) {
                closest = Some((agent, distance));
            }
        }

        if let Some((agent, _)) = closest {
            self.selected = Some(SelectedAgent::from_render_agent(agent));
            self.ui_state.selected_agent_id = Some(agent.entity_id);
            if let Err(error) = self.sync_selected_agent() {
                self.error_message = Some(error.to_string());
            }
        } else {
            self.selected = None;
            self.selected_data = None;
            self.ui_state.selected_agent_id = None;
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.update_fps();
        self.error_message = None;
        self.handle_shortcuts(ui.ctx(), _frame);
        if let Err(error) = self.drive_simulation() {
            self.error_message = Some(error.to_string());
            self.ui_state.paused = true;
        }

        self.draw_side_panel(ui);
        self.draw_world(ui);
        ui.ctx().request_repaint();
    }
}

fn refresh_snapshot(state: &mut SimulationState) -> Result<(UiStatsReadback, RenderSnapshotReadback)> {
    let ui_stats = state.ui_stats()?;
    let snapshot = state.render_snapshot(ui_stats.predator_count, ui_stats.prey_count, state.config().food_capacity)?;
    Ok((ui_stats, snapshot))
}

fn find_agent_by_entity(
    snapshot: &RenderSnapshotReadback,
    population_kind: PopulationKind,
    entity_id: u32,
) -> Option<&RenderAgentReadback> {
    let collection = match population_kind {
        PopulationKind::Predator => &snapshot.predators,
        PopulationKind::Prey => &snapshot.prey,
    };
    collection.iter().find(|agent| agent.entity_id == entity_id)
}

fn resolve_logo_path() -> Option<PathBuf> {
    let binary = std::env::current_exe().ok()?;
    let binary_dir = binary.parent()?;
    let candidate = binary_dir.join("logo.png");
    candidate.is_file().then_some(candidate)
}

fn load_icon() -> Option<egui::IconData> {
    let path = resolve_logo_path()?;
    let bytes = std::fs::read(&path).ok()?;
    eframe::icon_data::from_png_bytes(&bytes).ok()
}

fn screenshot_path(run_label: &str, tick: u32) -> Result<PathBuf> {
    let seconds = SystemTime::now().duration_since(UNIX_EPOCH).context("system time is before UNIX_EPOCH")?.as_secs();
    Ok(PathBuf::from("output").join("screenshots").join(format!("{run_label}_tick{tick}_{seconds}.png")))
}
