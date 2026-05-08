use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context as _, Result};
use eframe::egui::{self, Color32, ColorImage, Pos2, Rect, Stroke, StrokeKind, Vec2};
use image::ImageEncoder as _;

use crate::settings::UiConfig;
use crate::tick::buffers::{RenderAgentReadback, RenderFoodReadback, RenderSnapshotReadback};
use crate::ui::types::{CameraState, SelectedAgentData};

const GRID_DIVISIONS: u32 = 12;

pub struct SceneRenderInput<'a> {
    pub snapshot: &'a RenderSnapshotReadback,
    pub ui_config: &'a UiConfig,
    pub camera: CameraState,
    pub world_size: f32,
    pub selected: Option<&'a SelectedAgentData>,
    pub image_size: [usize; 2],
    pub vision_range: f32,
}

pub fn default_camera(world_size: f32) -> CameraState {
    CameraState::new(world_size * 0.5, world_size * 0.5, 1.0)
}

pub const fn clamp_camera(camera: &mut CameraState, world_size: f32, ui_config: &UiConfig) {
    camera.zoom = clamp_f32(camera.zoom, ui_config.zoom_min, ui_config.zoom_max);
    camera.center_x = clamp_f32(camera.center_x, 0.0, world_size);
    camera.center_y = clamp_f32(camera.center_y, 0.0, world_size);
}

pub fn world_to_screen(rect: Rect, camera: CameraState, world_size: f32, x: f32, y: f32) -> Pos2 {
    let scale = pixels_per_world(rect.size(), camera, world_size);
    let center = rect.center();
    Pos2::new(center.x + ((x - camera.center_x) * scale), center.y - ((y - camera.center_y) * scale))
}

pub fn screen_to_world(rect: Rect, camera: CameraState, world_size: f32, point: Pos2) -> (f32, f32) {
    let scale = pixels_per_world(rect.size(), camera, world_size);
    let center = rect.center();
    let world_x = camera.center_x + ((point.x - center.x) / scale);
    let world_y = camera.center_y - ((point.y - center.y) / scale);
    (world_x, world_y)
}

pub fn save_png(path: &Path, image: &ColorImage) -> Result<()> {
    let mut rgba = Vec::with_capacity(image.pixels.len() * 4);
    for pixel in &image.pixels {
        rgba.extend_from_slice(&[pixel.r(), pixel.g(), pixel.b(), pixel.a()]);
    }
    let file =
        std::fs::File::create(path).with_context(|| format!("failed to create screenshot at {}", path.display()))?;
    let writer = std::io::BufWriter::new(file);
    image::codecs::png::PngEncoder::new(writer)
        .write_image(
            &rgba,
            u32::try_from(image.size[0]).context("screenshot width overflowed")?,
            u32::try_from(image.size[1]).context("screenshot height overflowed")?,
            image::ColorType::Rgba8.into(),
        )
        .with_context(|| format!("failed to encode screenshot at {}", path.display()))
}

pub fn render_scene(input: &SceneRenderInput<'_>) -> ColorImage {
    let [width, height] = input.image_size;
    let mut raster = Raster::new(width.max(1), height.max(1), color_with_alpha(input.ui_config.bg_color, 255));

    draw_grid(&mut raster, input.ui_config, input.camera, input.world_size);
    draw_food(&mut raster, input.snapshot.food.as_slice(), input.ui_config, input.camera, input.world_size);
    draw_agents(
        &mut raster,
        input.snapshot.prey.as_slice(),
        input.ui_config.prey_color,
        input.ui_config.prey_radius,
        input.ui_config,
        input.camera,
        input.world_size,
    );
    draw_agents(
        &mut raster,
        input.snapshot.predators.as_slice(),
        input.ui_config.predator_color,
        input.ui_config.predator_radius,
        input.ui_config,
        input.camera,
        input.world_size,
    );
    if let Some(selected) = input.selected {
        draw_selected_overlay(
            &mut raster,
            selected,
            input.ui_config,
            input.camera,
            input.world_size,
            input.vision_range,
        );
    }

    raster.into_image()
}

pub fn paint_network(ui: &mut egui::Ui, rect: Rect, ui_config: &UiConfig, selected: &SelectedAgentData) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 8.0, color_with_alpha(ui_config.panel_bg_color, alpha_u8(ui_config.panel_alpha)));
    painter.rect_stroke(
        rect,
        8.0,
        Stroke::new(1.0, color_with_alpha(ui_config.panel_outline_color, 255)),
        StrokeKind::Outside,
    );

    let content = rect.shrink(14.0);
    let positions = node_positions(content, &selected.genome);

    for connection in &selected.genome.connections {
        let Some(from) = positions.get(&(connection.from_node as u32)) else {
            continue;
        };
        let Some(to) = positions.get(&(connection.to_node as u32)) else {
            continue;
        };
        let color = connection_color(connection.weight, connection.enabled != 0);
        let width = if connection.enabled != 0 { 1.5 } else { 0.75 };
        painter.line_segment([*from, *to], Stroke::new(width, color));
    }

    for node in &selected.genome.nodes {
        let Some(position) = positions.get(&node.id) else {
            continue;
        };
        let fill = node_color(ui_config, node.node_type);
        painter.circle_filled(*position, 8.0, fill);
        painter.circle_stroke(
            *position,
            8.0,
            Stroke::new(
                1.0,
                color_with_alpha(
                    ui_config.nn_node_outline_color_rgb(),
                    alpha_from_unit(ui_config.nn_node_outline_color[3]),
                ),
            ),
        );
    }
}

fn draw_grid(raster: &mut Raster, ui_config: &UiConfig, camera: CameraState, world_size: f32) {
    let border = color_with_alpha(ui_config.border_color, 255);
    let grid = color_with_alpha(ui_config.grid_color, 255);

    for index in 1..GRID_DIVISIONS {
        let world = world_size * (index as f32 / GRID_DIVISIONS as f32);
        let vertical_top = project_to_image(raster.size(), camera, world_size, world, world_size);
        let vertical_bottom = project_to_image(raster.size(), camera, world_size, world, 0.0);
        raster.draw_line(vertical_top, vertical_bottom, grid);

        let horizontal_left = project_to_image(raster.size(), camera, world_size, 0.0, world);
        let horizontal_right = project_to_image(raster.size(), camera, world_size, world_size, world);
        raster.draw_line(horizontal_left, horizontal_right, grid);
    }

    let bottom_left = project_to_image(raster.size(), camera, world_size, 0.0, 0.0);
    let bottom_right = project_to_image(raster.size(), camera, world_size, world_size, 0.0);
    let top_right = project_to_image(raster.size(), camera, world_size, world_size, world_size);
    let top_left = project_to_image(raster.size(), camera, world_size, 0.0, world_size);
    raster.draw_line(top_left, top_right, border);
    raster.draw_line(top_right, bottom_right, border);
    raster.draw_line(bottom_right, bottom_left, border);
    raster.draw_line(bottom_left, top_left, border);
}

fn draw_food(
    raster: &mut Raster,
    food: &[RenderFoodReadback],
    ui_config: &UiConfig,
    camera: CameraState,
    world_size: f32,
) {
    let color = color_with_alpha(ui_config.food_color, alpha_u8(ui_config.food_alpha));
    let radius = scaled_radius(raster.size(), camera, world_size, ui_config.food_radius).max(1.0);

    for entry in food {
        if entry.active == 0 {
            continue;
        }
        let (x, y) = project_to_image(raster.size(), camera, world_size, entry.pos_x, entry.pos_y);
        raster.draw_filled_circle(x, y, radius, color);
    }
}

fn draw_agents(
    raster: &mut Raster,
    agents: &[RenderAgentReadback],
    color: [f32; 3],
    radius_world: f32,
    ui_config: &UiConfig,
    camera: CameraState,
    world_size: f32,
) {
    let fill = color_with_alpha(color, 255);
    let radius = scaled_radius(raster.size(), camera, world_size, radius_world).max(2.0);

    for agent in agents {
        let (x, y) = project_to_image(raster.size(), camera, world_size, agent.pos_x, agent.pos_y);
        let direction = normalized_dir(agent.dir_x, agent.dir_y);
        let tip = (
            x + (direction.0 * radius * ui_config.triangle_tip_factor),
            y - (direction.1 * radius * ui_config.triangle_tip_factor),
        );
        let base_center = (
            x - (direction.0 * radius * ui_config.triangle_base_factor),
            y + (direction.1 * radius * ui_config.triangle_base_factor),
        );
        let perpendicular = (-direction.1, direction.0);
        let half_width = radius * ui_config.triangle_width_factor;
        let left = (base_center.0 + (perpendicular.0 * half_width), base_center.1 - (perpendicular.1 * half_width));
        let right = (base_center.0 - (perpendicular.0 * half_width), base_center.1 + (perpendicular.1 * half_width));
        raster.draw_filled_triangle(tip, left, right, fill);
    }
}

fn draw_selected_overlay(
    raster: &mut Raster,
    selected: &SelectedAgentData,
    ui_config: &UiConfig,
    camera: CameraState,
    world_size: f32,
    vision_range: f32,
) {
    let (x, y) = project_to_image(raster.size(), camera, world_size, selected.agent.pos_x, selected.agent.pos_y);
    let selection_radius =
        scaled_radius(raster.size(), camera, world_size, ui_config.predator_radius.max(ui_config.prey_radius))
            + ui_config.selected_outline_thickness
            + 2.0;
    raster.draw_circle_outline(x, y, selection_radius, color_with_alpha([1.0, 1.0, 1.0], 220));

    let vision_pixels = scaled_radius(raster.size(), camera, world_size, vision_range).max(2.0);
    raster.draw_circle_outline(
        x,
        y,
        vision_pixels,
        color_with_alpha(ui_config.vision_fill_rgb(), alpha_u8(ui_config.vision_outline_alpha)),
    );

    for line in sensor_lines(selected, vision_range) {
        let (x1, y1) = project_to_image(raster.size(), camera, world_size, line.0, line.1);
        let alpha = if line.2 == 2 { alpha_u8(ui_config.food_sensor_alpha) } else { alpha_u8(ui_config.sensor_alpha) };
        let color = match line.2 {
            0 => color_with_alpha(ui_config.predator_color, alpha),
            1 => color_with_alpha(ui_config.prey_color, alpha),
            _ => color_with_alpha(ui_config.food_color, alpha),
        };
        raster.draw_line((x, y), (x1, y1), color);
    }
}

fn sensor_lines(selected: &SelectedAgentData, vision_range: f32) -> Vec<(f32, f32, u8)> {
    let mut lines = Vec::with_capacity(15);
    for group in 0..3 {
        let start = group * 10;
        for pair in 0..5 {
            let dx = selected.sensors.inputs[start + (pair * 2)];
            let dy = selected.sensors.inputs[start + (pair * 2) + 1];
            if dx == 0.0 && dy == 0.0 {
                continue;
            }
            lines.push((
                selected.agent.pos_x + (dx * vision_range),
                selected.agent.pos_y + (dy * vision_range),
                group as u8,
            ));
        }
    }
    lines
}

fn node_positions(rect: Rect, genome: &crate::tick::species::RepresentativeGenomeReadback) -> HashMap<u32, Pos2> {
    let mut inputs = Vec::new();
    let mut hidden = Vec::new();
    let mut outputs = Vec::new();
    let mut bias = Vec::new();

    for node in &genome.nodes {
        match node.node_type {
            0 => inputs.push(node.id),
            1 => hidden.push(node.id),
            2 => outputs.push(node.id),
            3 => bias.push(node.id),
            _ => hidden.push(node.id),
        }
    }

    inputs.sort_unstable();
    hidden.sort_unstable();
    outputs.sort_unstable();
    bias.sort_unstable();

    let x_positions = [
        rect.left() + 20.0,
        rect.left() + (rect.width() * 0.33),
        rect.left() + (rect.width() * 0.66),
        rect.right() - 20.0,
    ];
    let mut positions = HashMap::new();
    insert_layer_positions(&mut positions, &inputs, x_positions[0], rect.top(), rect.bottom());
    insert_layer_positions(&mut positions, &bias, x_positions[1], rect.top(), rect.bottom());
    insert_layer_positions(&mut positions, &hidden, x_positions[2], rect.top(), rect.bottom());
    insert_layer_positions(&mut positions, &outputs, x_positions[3], rect.top(), rect.bottom());
    positions
}

fn insert_layer_positions(positions: &mut HashMap<u32, Pos2>, nodes: &[u32], x: f32, top: f32, bottom: f32) {
    if nodes.is_empty() {
        return;
    }

    let spacing = (bottom - top) / (nodes.len() as f32 + 1.0);
    for (index, id) in nodes.iter().enumerate() {
        let y = top + ((index as f32 + 1.0) * spacing);
        positions.insert(*id, Pos2::new(x, y));
    }
}

fn node_color(ui_config: &UiConfig, node_type: u8) -> Color32 {
    match node_type {
        0 => color_with_alpha(ui_config.nn_input_color, 255),
        1 => color_with_alpha(ui_config.nn_hidden_color, 255),
        2 => color_with_alpha(ui_config.nn_output_color, 255),
        3 => color_with_alpha(ui_config.nn_bias_color, 255),
        _ => color_with_alpha(ui_config.muted_color, 255),
    }
}

fn connection_color(weight: f32, enabled: bool) -> Color32 {
    if !enabled {
        return Color32::from_rgba_unmultiplied(110, 110, 110, 100);
    }

    let magnitude = weight.abs().clamp(0.0, 1.0);
    if weight >= 0.0 {
        let channel = (90.0 + (150.0 * magnitude)) as u8;
        Color32::from_rgb(64, channel, 255)
    } else {
        let channel = (90.0 + (150.0 * magnitude)) as u8;
        Color32::from_rgb(255, channel, 48)
    }
}

fn pixels_per_world(size: Vec2, camera: CameraState, world_size: f32) -> f32 {
    let shortest = size.x.min(size.y).max(1.0);
    (shortest / world_size.max(1.0)) * camera.zoom
}

fn project_to_image(size: [usize; 2], camera: CameraState, world_size: f32, x: f32, y: f32) -> (f32, f32) {
    let width = size[0] as f32;
    let height = size[1] as f32;
    let scale = pixels_per_world(Vec2::new(width, height), camera, world_size);
    let cx = width * 0.5;
    let cy = height * 0.5;
    (cx + ((x - camera.center_x) * scale), cy - ((y - camera.center_y) * scale))
}

fn scaled_radius(size: [usize; 2], camera: CameraState, world_size: f32, radius_world: f32) -> f32 {
    pixels_per_world(Vec2::new(size[0] as f32, size[1] as f32), camera, world_size) * radius_world
}

fn normalized_dir(x: f32, y: f32) -> (f32, f32) {
    let magnitude = (x * x + y * y).sqrt();
    if magnitude <= f32::EPSILON { (1.0, 0.0) } else { (x / magnitude, y / magnitude) }
}

fn color_with_alpha(color: [f32; 3], alpha: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(unit_to_u8(color[0]), unit_to_u8(color[1]), unit_to_u8(color[2]), alpha)
}

const fn alpha_u8(value: f32) -> u8 {
    if value <= 0.0 {
        0
    } else if value >= 255.0 {
        255
    } else {
        value as u8
    }
}

fn alpha_from_unit(value: f32) -> u8 {
    unit_to_u8(value)
}

fn unit_to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

struct Raster {
    width: usize,
    height: usize,
    pixels: Vec<Color32>,
}

impl Raster {
    fn new(width: usize, height: usize, fill: Color32) -> Self {
        Self { width, height, pixels: vec![fill; width * height] }
    }

    const fn size(&self) -> [usize; 2] {
        [self.width, self.height]
    }

    fn into_image(self) -> ColorImage {
        ColorImage::new([self.width, self.height], self.pixels)
    }

    fn draw_line(&mut self, start: (f32, f32), end: (f32, f32), color: Color32) {
        let dx = end.0 - start.0;
        let dy = end.1 - start.1;
        let steps = dx.abs().max(dy.abs()).max(1.0);
        let step_x = dx / steps;
        let step_y = dy / steps;
        let mut x = start.0;
        let mut y = start.1;
        for _ in 0..=steps as usize {
            self.blend_pixel(x.round() as i32, y.round() as i32, color);
            x += step_x;
            y += step_y;
        }
    }

    fn draw_filled_circle(&mut self, center_x: f32, center_y: f32, radius: f32, color: Color32) {
        let min_x = (center_x - radius).floor() as i32;
        let max_x = (center_x + radius).ceil() as i32;
        let min_y = (center_y - radius).floor() as i32;
        let max_y = (center_y + radius).ceil() as i32;
        let radius_sq = radius * radius;

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let dx = x as f32 - center_x;
                let dy = y as f32 - center_y;
                if (dx * dx) + (dy * dy) <= radius_sq {
                    self.blend_pixel(x, y, color);
                }
            }
        }
    }

    fn draw_circle_outline(&mut self, center_x: f32, center_y: f32, radius: f32, color: Color32) {
        let segments = 64;
        let mut previous = None;
        for index in 0..=segments {
            let theta = (index as f32 / segments as f32) * std::f32::consts::TAU;
            let point = (center_x + (theta.cos() * radius), center_y + (theta.sin() * radius));
            if let Some(last) = previous {
                self.draw_line(last, point, color);
            }
            previous = Some(point);
        }
    }

    fn draw_filled_triangle(&mut self, a: (f32, f32), b: (f32, f32), c: (f32, f32), color: Color32) {
        let min_x = a.0.min(b.0).min(c.0).floor() as i32;
        let max_x = a.0.max(b.0).max(c.0).ceil() as i32;
        let min_y = a.1.min(b.1).min(c.1).floor() as i32;
        let max_y = a.1.max(b.1).max(c.1).ceil() as i32;

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                if point_in_triangle((x as f32 + 0.5, y as f32 + 0.5), a, b, c) {
                    self.blend_pixel(x, y, color);
                }
            }
        }
    }

    fn blend_pixel(&mut self, x: i32, y: i32, color: Color32) {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return;
        }
        let index = y as usize * self.width + x as usize;
        let dst = self.pixels[index];
        self.pixels[index] = blend(dst, color);
    }
}

fn point_in_triangle(point: (f32, f32), a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> bool {
    let area = edge(a, b, c);
    if area == 0.0 {
        return false;
    }
    let w0 = edge(b, c, point);
    let w1 = edge(c, a, point);
    let w2 = edge(a, b, point);
    (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0) || (w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0)
}

fn edge(a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> f32 {
    ((c.0 - a.0) * (b.1 - a.1)) - ((c.1 - a.1) * (b.0 - a.0))
}

fn blend(dst: Color32, src: Color32) -> Color32 {
    let src_alpha = src.a() as f32 / 255.0;
    let dst_alpha = dst.a() as f32 / 255.0;
    let out_alpha = src_alpha + (dst_alpha * (1.0 - src_alpha));
    if out_alpha <= f32::EPSILON {
        return Color32::TRANSPARENT;
    }

    let out_r = ((src.r() as f32 * src_alpha) + (dst.r() as f32 * dst_alpha * (1.0 - src_alpha))) / out_alpha;
    let out_g = ((src.g() as f32 * src_alpha) + (dst.g() as f32 * dst_alpha * (1.0 - src_alpha))) / out_alpha;
    let out_b = ((src.b() as f32 * src_alpha) + (dst.b() as f32 * dst_alpha * (1.0 - src_alpha))) / out_alpha;

    Color32::from_rgba_unmultiplied(
        out_r.round() as u8,
        out_g.round() as u8,
        out_b.round() as u8,
        (out_alpha * 255.0).round() as u8,
    )
}

trait UiConfigColorExt {
    fn vision_fill_rgb(&self) -> [f32; 3];
    fn nn_node_outline_color_rgb(&self) -> [f32; 3];
}

impl UiConfigColorExt for UiConfig {
    fn vision_fill_rgb(&self) -> [f32; 3] {
        [self.vision_fill_color[0], self.vision_fill_color[1], self.vision_fill_color[2]]
    }

    fn nn_node_outline_color_rgb(&self) -> [f32; 3] {
        [self.nn_node_outline_color[0], self.nn_node_outline_color[1], self.nn_node_outline_color[2]]
    }
}

const fn clamp_f32(value: f32, min: f32, max: f32) -> f32 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vision_radius_matches_world_projection_distance() {
        let size = [1280, 720];
        let camera = CameraState::new(1800.0, 1800.0, 1.75);
        let world_size = 3600.0;
        let vision_range = 128.0;
        let center = project_to_image(size, camera, world_size, 1400.0, 1400.0);
        let edge = project_to_image(size, camera, world_size, 1528.0, 1400.0);
        let rendered_radius = scaled_radius(size, camera, world_size, vision_range);
        let projected_distance = (edge.0 - center.0).abs();

        assert!((rendered_radius - projected_distance).abs() <= 0.001);
    }
}
