use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context as _, Result};
use eframe::egui::{self, Color32, Key, Pos2, Sense, Shape, Stroke, Vec2};

use crate::config::SimulationConfig;
use crate::profile_scope;
use crate::profiler::Profiler;
use crate::settings::UiConfig;
use crate::tick::buffers::{MetricsSummaryReadback, RenderAgentReadback, RenderSnapshotReadback, UiStatsReadback};
use crate::tick::simulation::PopulationKind;
use crate::tick::simulation::SimulationState;
use crate::ui::render;
use crate::ui::types::{
    CameraState, OverlayHistory, OverlayStats, PairHistoryPoint, PopulationHistoryPoint, SelectedAgent,
    SelectedAgentData, UiState,
};
use crate::ui::world::{self, WorldFrame};

pub struct App {
    run_label: String,
    config: SimulationConfig,
    ui_config: UiConfig,
    state: SimulationState,
    ui_state: UiState,
    camera: CameraState,
    ui_stats: UiStatsReadback,
    metrics_summary: MetricsSummaryReadback,
    world_frame: Arc<WorldFrame>,
    overlay_history: OverlayHistory,
    profiler: Profiler,
    selected: Option<SelectedAgent>,
    selected_data: Option<SelectedAgentData>,
    last_frame_started: Instant,
    fps: f32,
    status_message: Option<String>,
    error_message: Option<String>,
    reached_tick_limit: bool,
}

impl App {
    pub fn run(run_label: &str, config: &SimulationConfig, ui_config: &UiConfig) -> Result<()> {
        let native_options = eframe::NativeOptions {
            renderer: eframe::Renderer::Wgpu,
            viewport: egui::ViewportBuilder::default()
                .with_title(format!("MoonAI - {run_label}"))
                .with_inner_size([ui_config.window_width as f32, ui_config.window_height as f32]),
            ..Default::default()
        };

        let run_label = run_label.to_owned();
        let config_for_app = *config;
        let ui_for_app = ui_config.clone();
        eframe::run_native(
            "MoonAI",
            native_options,
            Box::new(move |_creation_context| {
                Self::new(_creation_context, &run_label, config_for_app, ui_for_app.clone())
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
        run_label: &str,
        config: SimulationConfig,
        ui_config: UiConfig,
    ) -> Result<Self> {
        let render_state = creation_context
            .wgpu_render_state
            .as_ref()
            .context("eframe did not provide a wgpu render state for the GPU world renderer")?;
        world::install_renderer_resources(render_state, &ui_config)?;

        let mut state = SimulationState::init_from_config(&config)?;
        let camera = render::default_camera(config.grid_size);
        let initial_ui_stats = state.ui_stats()?;
        let (ui_stats, metrics_summary, world_frame) = load_world_frame(&mut state, initial_ui_stats, &ui_config)?;
        let initial_overlay =
            OverlayStats::from_snapshot(ui_stats, metrics_summary, world_frame.active_food_count(), 1, false, 0.0);
        let mut overlay_history = OverlayHistory::default();
        overlay_history.push(&initial_overlay);

        Ok(Self {
            run_label: run_label.to_owned(),
            config,
            ui_config,
            state,
            ui_state: UiState::default(),
            camera,
            ui_stats,
            metrics_summary,
            world_frame,
            overlay_history,
            profiler: Profiler::default(),
            selected: None,
            selected_data: None,
            last_frame_started: Instant::now(),
            fps: 0.0,
            status_message: None,
            error_message: None,
            reached_tick_limit: false,
        })
    }

    fn tick_limit(&self) -> Option<u32> {
        (self.config.max_ticks > 0).then_some(self.config.max_ticks)
    }

    fn handle_shortcuts(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        profile_scope!("handle_events");

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
            self.camera = render::default_camera(self.config.grid_size);
        }
        if ctx.input(|input| input.key_pressed(Key::Escape)) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
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
        if steps == 0 {
            self.ui_state.tick_requested = false;
            return Ok(());
        }

        let limit = self.tick_limit();
        let mut advanced = false;
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
            advanced = true;
        }
        self.ui_state.tick_requested = false;
        if !advanced {
            return Ok(());
        }

        let (ui_stats, metrics_summary, world_frame) =
            load_world_frame(&mut self.state, self.ui_stats, &self.ui_config)?;
        self.ui_stats = ui_stats;
        self.metrics_summary = metrics_summary;
        self.world_frame = world_frame;
        {
            profile_scope!("history");
            let overlay = self.overlay_stats();
            self.overlay_history.push(&overlay);
        }
        self.sync_selected_agent()?;
        Ok(())
    }

    fn overlay_stats(&self) -> OverlayStats {
        OverlayStats::from_snapshot(
            self.ui_stats,
            self.metrics_summary,
            self.world_frame.active_food_count(),
            self.ui_state.speed_multiplier,
            self.ui_state.paused,
            self.fps,
        )
    }

    fn sync_selected_agent(&mut self) -> Result<()> {
        profile_scope!("selected_agent");

        let Some(selected) = self.selected else {
            self.selected_data = None;
            self.ui_state.selected_agent_id = None;
            return Ok(());
        };

        let Some(agent) =
            find_agent_by_entity(self.world_frame.snapshot.as_ref(), selected.population_kind, selected.entity_id)
        else {
            self.selected = None;
            self.selected_data = None;
            self.ui_state.selected_agent_id = None;
            return Ok(());
        };

        let resolved = SelectedAgent::from_render_agent(agent);
        self.selected = Some(resolved);
        self.ui_state.selected_agent_id = Some(resolved.entity_id);
        let sensors = self.state.sensor_snapshot(agent.population_kind, agent.slot)?;
        let network = self.state.selected_agent_network(agent.population_kind, agent.slot)?;
        match &mut self.selected_data {
            Some(data) if data.agent.entity_id == agent.entity_id => {
                data.agent = *agent;
                data.sensors = sensors;
                data.network = network;
            }
            _ => {
                self.selected_data = Some(SelectedAgentData {
                    agent: *agent,
                    sensors,
                    network,
                    genome: self.state.representative_genome(agent.population_kind, agent.slot)?,
                });
            }
        }
        Ok(())
    }

    fn update_fps(&mut self) {
        profile_scope!("metrics");

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
        profile_scope!("side_panel");

        egui::Panel::left("moonai_left_panel")
            .resizable(false)
            .default_size(self.ui_config.ui_side_margin)
            .show_inside(ui, |ui| {
                let overlay = self.overlay_stats();
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    ui.heading("MoonAI");
                    ui.label(format!("Experiment: {}", self.run_label));

                    ui.separator();
                    ui.label(format!("Tick: {}", overlay.ui_stats.tick));
                    ui.label(format!("FPS: {:.1}", overlay.fps));
                    ui.label(format!("Speed: {}x", overlay.speed_multiplier));
                    ui.colored_label(
                        if overlay.paused { rgb(self.ui_config.pause_color) } else { rgb(self.ui_config.muted_color) },
                        if overlay.paused { "State: paused" } else { "State: running" },
                    );

                    ui.separator();
                    ui.label("Controls");
                    ui.label("Space: pause/resume");
                    ui.label("Up/Down or +/-: speed");
                    ui.label(".: single tick while paused");
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

                    ui.separator();
                    ui.heading("Profiler");
                    let profiler_rows = self.profiler.formatted_rows("frame", overlay.ui_stats.tick);
                    if profiler_rows.is_empty() {
                        ui.label("Warming up...");
                    } else {
                        for row in profiler_rows {
                            ui.monospace(row);
                        }
                    }

                    if let Some(selected) = &self.selected_data {
                        ui.separator();
                        ui.heading("Selected Agent");
                        ui.label(format!("Population: {:?}", selected.agent.population_kind));
                        ui.label(format!("Entity: {}", selected.agent.entity_id));
                        ui.label(format!("Slot: {}", selected.agent.slot));
                        ui.label(format!("Species: {}", selected.agent.species_id));
                        ui.label(format!("Generation: {}", selected.agent.generation));
                        ui.label(format!("Age: {:.0}", selected.agent.age));
                        ui.label(format!("Energy: {:.3}", selected.agent.energy));
                        ui.label(format!(
                            "Complexity: {}",
                            usize::from(selected.genome.header.num_nodes)
                                + usize::from(selected.genome.header.num_connections)
                        ));
                        ui.label(format!(
                            "Outputs: {:.3}, {:.3}",
                            selected.network.output_0, selected.network.output_1
                        ));
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
            });
    }

    fn draw_render(&mut self, ui: &mut egui::Ui) {
        profile_scope!("render");

        self.draw_side_panel(ui);
        self.draw_world(ui);
    }

    fn draw_world(&mut self, ui: &mut egui::Ui) {
        profile_scope!("world");

        egui::Panel::right("moonai_right_panel")
            .resizable(false)
            .default_size(self.ui_config.ui_side_margin)
            .show_inside(ui, |ui| {
                let overlay = self.overlay_stats();
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    ui.heading("Stats");
                    colored_stat(
                        ui,
                        self.ui_config.predator_color,
                        "Predators",
                        overlay.ui_stats.predator_count.to_string(),
                    );
                    colored_stat(ui, self.ui_config.prey_color, "Prey", overlay.ui_stats.prey_count.to_string());
                    colored_stat(ui, self.ui_config.food_color, "Food", overlay.active_food_count.to_string());
                    ui.separator();

                    ui.colored_label(rgb(self.ui_config.predator_color), "Predator");
                    ui.label(format!("Species: {}", overlay.metrics_summary.predator_species));
                    ui.label(format!("Energy: {:.2}", overlay.ui_stats.avg_predator_energy));
                    ui.label(format!("Complexity: {:.1}", overlay.metrics_summary.avg_predator_complexity));
                    ui.label(format!("Births: {}", overlay.ui_stats.predator_births));
                    ui.label(format!("Deaths: {}", overlay.ui_stats.predator_deaths));
                    ui.label(format!(
                        "Generation: max={} avg={:.1}",
                        overlay.metrics_summary.max_predator_generation,
                        overlay.metrics_summary.avg_predator_generation
                    ));
                    ui.label(format!("Kills: {}", overlay.ui_stats.kills));

                    ui.separator();
                    ui.colored_label(rgb(self.ui_config.prey_color), "Prey");
                    ui.label(format!("Species: {}", overlay.metrics_summary.prey_species));
                    ui.label(format!("Energy: {:.2}", overlay.ui_stats.avg_prey_energy));
                    ui.label(format!("Complexity: {:.1}", overlay.metrics_summary.avg_prey_complexity));
                    ui.label(format!("Births: {}", overlay.ui_stats.prey_births));
                    ui.label(format!("Deaths: {}", overlay.ui_stats.prey_deaths));
                    ui.label(format!(
                        "Generation: max={} avg={:.1}",
                        overlay.metrics_summary.max_prey_generation, overlay.metrics_summary.avg_prey_generation
                    ));
                    ui.label(format!("Food eaten: {}", overlay.ui_stats.food_eaten));

                    ui.separator();
                    draw_chart_panel(
                        ui,
                        &self.ui_config,
                        "Population",
                        &[
                            ChartSeries::population(
                                &self.overlay_history.population,
                                self.ui_config.predator_color,
                                |point| point.predators as f32,
                            ),
                            ChartSeries::population(
                                &self.overlay_history.population,
                                self.ui_config.prey_color,
                                |point| point.prey as f32,
                            ),
                            ChartSeries::population(
                                &self.overlay_history.population,
                                self.ui_config.food_color,
                                |point| point.food as f32,
                            ),
                        ],
                        180.0,
                    );
                    draw_chart_panel(
                        ui,
                        &self.ui_config,
                        "Complexity",
                        &[
                            ChartSeries::pair(
                                &self.overlay_history.complexity,
                                self.ui_config.predator_color,
                                |point| point.predator,
                            ),
                            ChartSeries::pair(&self.overlay_history.complexity, self.ui_config.prey_color, |point| {
                                point.prey
                            }),
                        ],
                        120.0,
                    );
                    draw_chart_panel(
                        ui,
                        &self.ui_config,
                        "Energy",
                        &[
                            ChartSeries::pair(&self.overlay_history.energy, self.ui_config.predator_color, |point| {
                                point.predator
                            }),
                            ChartSeries::pair(&self.overlay_history.energy, self.ui_config.prey_color, |point| {
                                point.prey
                            }),
                        ],
                        120.0,
                    );
                });
            });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            let (rect, response) = world::allocate_world_rect(ui);
            self.handle_view_input(ui.ctx(), response.rect, &response);
            world::paint_world(
                ui,
                rect,
                Arc::clone(&self.world_frame),
                &self.ui_config,
                self.camera,
                self.config.grid_size,
                self.selected_data.as_ref(),
                self.config.vision_range,
            );
        });
    }

    fn handle_view_input(&mut self, ctx: &egui::Context, rect: egui::Rect, response: &egui::Response) {
        profile_scope!("input");

        if response.hovered() {
            let scroll = ctx.input(|input| input.smooth_scroll_delta.y);
            if scroll.abs() > f32::EPSILON {
                let hover = ctx.input(|input| input.pointer.hover_pos()).unwrap_or(rect.center());
                let world_before = render::screen_to_world(rect, self.camera, self.config.grid_size, hover);
                let zoom_factor = (scroll * 0.0015).exp();
                self.camera.zoom *= zoom_factor;
                render::clamp_camera(&mut self.camera, self.config.grid_size, &self.ui_config);
                let world_after = render::screen_to_world(rect, self.camera, self.config.grid_size, hover);
                self.camera.center_x += world_before.0 - world_after.0;
                self.camera.center_y += world_before.1 - world_after.1;
                render::clamp_camera(&mut self.camera, self.config.grid_size, &self.ui_config);
            }
        }

        let dragging =
            response.dragged_by(egui::PointerButton::Middle) || response.dragged_by(egui::PointerButton::Secondary);
        if dragging {
            let delta = ctx.input(|input| input.pointer.delta());
            let scale =
                rect.width().min(rect.height()).max(1.0) / (self.config.grid_size / self.camera.zoom.max(0.001));
            self.camera.center_x -= delta.x / scale;
            self.camera.center_y += delta.y / scale;
            render::clamp_camera(&mut self.camera, self.config.grid_size, &self.ui_config);
        }

        if response.clicked_by(egui::PointerButton::Primary)
            && let Some(pointer) = response.interact_pointer_pos()
        {
            self.select_agent(rect, pointer);
        }
    }

    fn select_agent(&mut self, rect: egui::Rect, pointer: egui::Pos2) {
        profile_scope!("selection");

        let mut closest: Option<(&RenderAgentReadback, f32)> = None;
        let select_radius = self.ui_config.selection_click_radius;

        for agent in self.world_frame.snapshot.predators.iter().chain(self.world_frame.snapshot.prey.iter()) {
            let screen = render::world_to_screen(rect, self.camera, self.config.grid_size, agent.pos_x, agent.pos_y);
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
        let _profiler_session = self.profiler.bind();
        profile_scope!("frame");

        self.update_fps();
        self.error_message = None;
        self.handle_shortcuts(ui.ctx(), _frame);
        if let Err(error) = self.drive_simulation() {
            self.error_message = Some(error.to_string());
            self.ui_state.paused = true;
        }

        self.draw_render(ui);
        if !self.ui_state.paused {
            ui.ctx().request_repaint();
        }
    }
}

fn load_world_frame(
    state: &mut SimulationState,
    ui_stats: UiStatsReadback,
    ui_config: &UiConfig,
) -> Result<(UiStatsReadback, MetricsSummaryReadback, Arc<WorldFrame>)> {
    profile_scope!("world_frame");

    let metrics_summary = state.metrics_summary()?;
    let snapshot = state.render_snapshot(ui_stats.predator_count, ui_stats.prey_count, state.config().food_count)?;
    let world_frame = Arc::new(WorldFrame::from_snapshot(snapshot, ui_config));
    Ok((ui_stats, metrics_summary, world_frame))
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

struct ChartSeries<'a> {
    values: Vec<f32>,
    color: Color32,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> ChartSeries<'a> {
    fn population(
        points: &std::collections::VecDeque<PopulationHistoryPoint>,
        color: [f32; 3],
        map: impl Fn(&PopulationHistoryPoint) -> f32,
    ) -> Self {
        Self { values: points.iter().map(map).collect(), color: rgb(color), _marker: std::marker::PhantomData }
    }

    fn pair(
        points: &std::collections::VecDeque<PairHistoryPoint>,
        color: [f32; 3],
        map: impl Fn(&PairHistoryPoint) -> f32,
    ) -> Self {
        Self { values: points.iter().map(map).collect(), color: rgb(color), _marker: std::marker::PhantomData }
    }
}

fn colored_stat(ui: &mut egui::Ui, color: [f32; 3], label: &str, value: String) {
    ui.horizontal(|ui| {
        ui.colored_label(rgb(color), label);
        ui.label(value);
    });
}

fn draw_chart_panel(ui: &mut egui::Ui, ui_config: &UiConfig, title: &str, series: &[ChartSeries<'_>], height: f32) {
    ui.label(title);
    let width = ui.available_width().max(1.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 8.0, panel_fill(ui_config));
    painter.rect_stroke(rect, 8.0, Stroke::new(1.0, rgb(ui_config.panel_outline_color)), egui::StrokeKind::Outside);

    let content = rect.shrink2(Vec2::new(8.0, 12.0));
    let max_value = series.iter().flat_map(|entry| entry.values.iter().copied()).fold(1.0_f32, f32::max).max(1.0);

    for entry in series {
        if entry.values.len() < 2 {
            continue;
        }

        let last_index = (entry.values.len() - 1) as f32;
        let points: Vec<Pos2> = entry
            .values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let x = if last_index <= f32::EPSILON {
                    content.left()
                } else {
                    content.left() + ((index as f32 / last_index) * content.width())
                };
                let y = content.bottom() - ((value / max_value) * content.height());
                Pos2::new(x, y)
            })
            .collect();
        painter.add(Shape::line(points, Stroke::new(1.5, entry.color)));
    }
}

fn rgb(color: [f32; 3]) -> Color32 {
    Color32::from_rgb(
        (color[0].clamp(0.0, 1.0) * 255.0) as u8,
        (color[1].clamp(0.0, 1.0) * 255.0) as u8,
        (color[2].clamp(0.0, 1.0) * 255.0) as u8,
    )
}

fn panel_fill(ui_config: &UiConfig) -> Color32 {
    Color32::from_rgba_unmultiplied(
        (ui_config.panel_bg_color[0].clamp(0.0, 1.0) * 255.0) as u8,
        (ui_config.panel_bg_color[1].clamp(0.0, 1.0) * 255.0) as u8,
        (ui_config.panel_bg_color[2].clamp(0.0, 1.0) * 255.0) as u8,
        (ui_config.panel_alpha.clamp(0.0, 1.0) * 255.0) as u8,
    )
}
