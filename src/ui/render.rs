use std::collections::HashMap;

use eframe::egui::{self, Color32, Pos2, Rect, Stroke, StrokeKind, Vec2};

use crate::settings::UiConfig;
use crate::ui::types::{CameraState, SelectedAgentData};

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

fn node_positions(rect: Rect, genome: &crate::sim::RepresentativeGenomeReadback) -> HashMap<u32, Pos2> {
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

trait UiConfigColorExt {
    fn nn_node_outline_color_rgb(&self) -> [f32; 3];
}

impl UiConfigColorExt for UiConfig {
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
        let rect = Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 720.0));
        let camera = CameraState::new(1800.0, 1800.0, 1.75);
        let world_size = 3600.0;
        let vision_range = 128.0;
        let center = world_to_screen(rect, camera, world_size, 1400.0, 1400.0);
        let edge = world_to_screen(rect, camera, world_size, 1528.0, 1400.0);
        let projected_distance = (edge.x - center.x).abs();
        let rendered_radius = pixels_per_world(rect.size(), camera, world_size) * vision_range;

        assert!((rendered_radius - projected_distance).abs() <= 0.001);
    }
}
