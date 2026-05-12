use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SettingsError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Settings JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct AppSettings {
    #[serde(default)]
    pub ui: UiConfig,
}

pub fn load_settings(root_dir: &Path) -> Result<AppSettings, SettingsError> {
    let settings_path = root_dir.join("settings.json");
    let file = match std::fs::File::open(&settings_path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(AppSettings::default()),
        Err(error) => return Err(SettingsError::Io(error)),
    };
    Ok(serde_json::from_reader(file)?)
}

pub fn save_settings(root_dir: &Path, settings: &AppSettings) -> Result<(), SettingsError> {
    let settings_path = root_dir.join("settings.json");
    let file = std::fs::File::create(settings_path)?;
    serde_json::to_writer_pretty(std::io::BufWriter::new(file), settings)?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UiConfig {
    #[serde(default = "default_predator_size")]
    pub predator_size: f32,
    #[serde(default = "default_prey_size")]
    pub prey_size: f32,
    #[serde(default = "default_food_size")]
    pub food_size: f32,
    #[serde(default = "default_predator_color", with = "rgb_hex")]
    pub predator_color: [f32; 3],
    #[serde(default = "default_prey_color", with = "rgb_hex")]
    pub prey_color: [f32; 3],
    #[serde(default = "default_food_color", with = "rgb_hex")]
    pub food_color: [f32; 3],
    #[serde(default = "default_grid_color", with = "rgb_hex")]
    pub grid_color: [f32; 3],
    #[serde(default = "default_border_color", with = "rgb_hex")]
    pub border_color: [f32; 3],
    #[serde(default = "default_bg_color", with = "rgb_hex")]
    pub bg_color: [f32; 3],
    #[serde(default = "default_panel_bg_color", with = "rgb_hex")]
    pub panel_bg_color: [f32; 3],
    #[serde(default = "default_panel_alpha")]
    pub panel_alpha: f32,
    #[serde(default = "default_panel_outline_color", with = "rgb_hex")]
    pub panel_outline_color: [f32; 3],
    #[serde(default = "default_vision_outline_alpha")]
    pub vision_outline_alpha: f32,
    #[serde(default = "default_vision_outline_color", with = "rgb_hex")]
    pub vision_outline_color: [f32; 3],
    #[serde(default = "default_sensor_alpha")]
    pub sensor_alpha: f32,
    #[serde(default = "default_food_sensor_alpha")]
    pub food_sensor_alpha: f32,
    #[serde(default = "default_food_alpha")]
    pub food_alpha: f32,
    #[serde(default = "default_selected_outline_thickness")]
    pub selected_outline_thickness: f32,
    #[serde(default = "default_triangle_tip_factor")]
    pub triangle_tip_factor: f32,
    #[serde(default = "default_triangle_base_factor")]
    pub triangle_base_factor: f32,
    #[serde(default = "default_triangle_width_factor")]
    pub triangle_width_factor: f32,
    #[serde(default = "default_muted_color", with = "rgb_hex")]
    pub muted_color: [f32; 3],
    #[serde(default = "default_pause_color", with = "rgb_hex")]
    pub pause_color: [f32; 3],
    #[serde(default = "default_nn_node_outline_color", with = "rgba_hex")]
    pub nn_node_outline_color: [f32; 4],
    #[serde(default = "default_nn_input_color", with = "rgb_hex")]
    pub nn_input_color: [f32; 3],
    #[serde(default = "default_nn_bias_color", with = "rgb_hex")]
    pub nn_bias_color: [f32; 3],
    #[serde(default = "default_nn_hidden_color", with = "rgb_hex")]
    pub nn_hidden_color: [f32; 3],
    #[serde(default = "default_nn_output_color", with = "rgb_hex")]
    pub nn_output_color: [f32; 3],
    #[serde(default = "default_left_panel_width")]
    pub left_panel_width: f32,
    #[serde(default = "default_right_panel_width")]
    pub right_panel_width: f32,
    #[serde(default = "default_font_size")]
    pub font_size: f32,
    #[serde(default = "default_window_width")]
    pub window_width: u32,
    #[serde(default = "default_window_height")]
    pub window_height: u32,
    #[serde(default = "default_nn_panel_min_height")]
    pub nn_panel_min_height: f32,
    #[serde(default = "default_nn_panel_min_width")]
    pub nn_panel_min_width: f32,
    #[serde(default = "default_nn_panel_max_width")]
    pub nn_panel_max_width: f32,
    #[serde(default = "default_selection_click_radius")]
    pub selection_click_radius: f32,
    #[serde(default = "default_fps_alpha")]
    pub fps_alpha: f32,
    #[serde(default = "default_zoom_min")]
    pub zoom_min: f32,
    #[serde(default = "default_zoom_max")]
    pub zoom_max: f32,
    #[serde(default = "default_speed_min")]
    pub speed_min: u32,
    #[serde(default = "default_speed_max")]
    pub speed_max: u32,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            predator_size: default_predator_size(),
            prey_size: default_prey_size(),
            food_size: default_food_size(),
            predator_color: default_predator_color(),
            prey_color: default_prey_color(),
            food_color: default_food_color(),
            grid_color: default_grid_color(),
            border_color: default_border_color(),
            bg_color: default_bg_color(),
            panel_bg_color: default_panel_bg_color(),
            panel_alpha: default_panel_alpha(),
            panel_outline_color: default_panel_outline_color(),
            vision_outline_alpha: default_vision_outline_alpha(),
            vision_outline_color: default_vision_outline_color(),
            sensor_alpha: default_sensor_alpha(),
            food_sensor_alpha: default_food_sensor_alpha(),
            food_alpha: default_food_alpha(),
            selected_outline_thickness: default_selected_outline_thickness(),
            triangle_tip_factor: default_triangle_tip_factor(),
            triangle_base_factor: default_triangle_base_factor(),
            triangle_width_factor: default_triangle_width_factor(),
            muted_color: default_muted_color(),
            pause_color: default_pause_color(),
            nn_node_outline_color: default_nn_node_outline_color(),
            nn_input_color: default_nn_input_color(),
            nn_bias_color: default_nn_bias_color(),
            nn_hidden_color: default_nn_hidden_color(),
            nn_output_color: default_nn_output_color(),
            left_panel_width: default_left_panel_width(),
            right_panel_width: default_right_panel_width(),
            font_size: default_font_size(),
            window_width: default_window_width(),
            window_height: default_window_height(),
            nn_panel_min_height: default_nn_panel_min_height(),
            nn_panel_min_width: default_nn_panel_min_width(),
            nn_panel_max_width: default_nn_panel_max_width(),
            selection_click_radius: default_selection_click_radius(),
            fps_alpha: default_fps_alpha(),
            zoom_min: default_zoom_min(),
            zoom_max: default_zoom_max(),
            speed_min: default_speed_min(),
            speed_max: default_speed_max(),
        }
    }
}

const fn default_predator_size() -> f32 {
    1.2
}
const fn default_prey_size() -> f32 {
    1.0
}
const fn default_food_size() -> f32 {
    0.6
}
const fn default_predator_color() -> [f32; 3] {
    [1.0, 0.42, 0.21]
}
const fn default_prey_color() -> [f32; 3] {
    [0.306, 0.804, 0.769]
}
const fn default_food_color() -> [f32; 3] {
    [0.667, 1.0, 0.427]
}
const fn default_grid_color() -> [f32; 3] {
    [0.137, 0.125, 0.153]
}
const fn default_border_color() -> [f32; 3] {
    [0.306, 0.29, 0.325]
}
const fn default_bg_color() -> [f32; 3] {
    [0.0, 0.0, 0.0]
}
const fn default_panel_bg_color() -> [f32; 3] {
    [0.063, 0.051, 0.078]
}
const fn default_panel_alpha() -> f32 {
    120.0
}
const fn default_panel_outline_color() -> [f32; 3] {
    [0.137, 0.125, 0.153]
}
const fn default_vision_outline_alpha() -> f32 {
    40.0
}
const fn default_vision_outline_color() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}
const fn default_sensor_alpha() -> f32 {
    80.0
}
const fn default_food_sensor_alpha() -> f32 {
    60.0
}
const fn default_food_alpha() -> f32 {
    180.0
}
const fn default_selected_outline_thickness() -> f32 {
    2.0
}
const fn default_triangle_tip_factor() -> f32 {
    1.5
}
const fn default_triangle_base_factor() -> f32 {
    0.8
}
const fn default_triangle_width_factor() -> f32 {
    0.7
}
const fn default_muted_color() -> [f32; 3] {
    [0.706, 0.706, 0.706]
}
const fn default_pause_color() -> [f32; 3] {
    [1.0, 0.588, 0.392]
}
const fn default_nn_node_outline_color() -> [f32; 4] {
    [0.784, 0.784, 0.784, 0.471]
}
const fn default_nn_input_color() -> [f32; 3] {
    [0.314, 0.471, 0.863]
}
const fn default_nn_bias_color() -> [f32; 3] {
    [0.235, 0.706, 0.863]
}
const fn default_nn_hidden_color() -> [f32; 3] {
    [0.863, 0.784, 0.314]
}
const fn default_nn_output_color() -> [f32; 3] {
    [0.863, 0.314, 0.314]
}
const fn default_left_panel_width() -> f32 {
    300.0
}
const fn default_right_panel_width() -> f32 {
    300.0
}
const fn default_font_size() -> f32 {
    14.0
}
const fn default_window_width() -> u32 {
    1920
}
const fn default_window_height() -> u32 {
    1080
}
const fn default_nn_panel_min_height() -> f32 {
    320.0
}
const fn default_nn_panel_min_width() -> f32 {
    340.0
}
const fn default_nn_panel_max_width() -> f32 {
    420.0
}
const fn default_selection_click_radius() -> f32 {
    60.0
}
const fn default_fps_alpha() -> f32 {
    0.1
}
const fn default_zoom_min() -> f32 {
    0.1
}
const fn default_zoom_max() -> f32 {
    10.0
}
const fn default_speed_min() -> u32 {
    1
}
const fn default_speed_max() -> u32 {
    1024
}

mod rgb_hex {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(color: &[f32; 3], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&format!(
            "#{:02X}{:02X}{:02X}",
            super::unit_to_u8(color[0]),
            super::unit_to_u8(color[1]),
            super::unit_to_u8(color[2])
        ))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[f32; 3], D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        super::parse_hex_rgb(&value).map_err(serde::de::Error::custom)
    }
}

mod rgba_hex {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(color: &[f32; 4], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&format!(
            "#{:02X}{:02X}{:02X}{:02X}",
            super::unit_to_u8(color[0]),
            super::unit_to_u8(color[1]),
            super::unit_to_u8(color[2]),
            super::unit_to_u8(color[3])
        ))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[f32; 4], D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        super::parse_hex_rgba(&value).map_err(serde::de::Error::custom)
    }
}

fn parse_hex_rgb(value: &str) -> Result<[f32; 3], String> {
    let hex = value.trim().strip_prefix('#').unwrap_or(value.trim());
    if hex.len() != 6 {
        return Err(format!("expected 6 hex digits for RGB color, got '{value}'"));
    }
    Ok([
        u8_to_unit(parse_hex_byte(&hex[0..2], value)?),
        u8_to_unit(parse_hex_byte(&hex[2..4], value)?),
        u8_to_unit(parse_hex_byte(&hex[4..6], value)?),
    ])
}

fn parse_hex_rgba(value: &str) -> Result<[f32; 4], String> {
    let hex = value.trim().strip_prefix('#').unwrap_or(value.trim());
    if hex.len() != 8 {
        return Err(format!("expected 8 hex digits for RGBA color, got '{value}'"));
    }
    Ok([
        u8_to_unit(parse_hex_byte(&hex[0..2], value)?),
        u8_to_unit(parse_hex_byte(&hex[2..4], value)?),
        u8_to_unit(parse_hex_byte(&hex[4..6], value)?),
        u8_to_unit(parse_hex_byte(&hex[6..8], value)?),
    ])
}

fn parse_hex_byte(value: &str, full: &str) -> Result<u8, String> {
    u8::from_str_radix(value, 16).map_err(|_| format!("invalid hex color '{full}'"))
}

fn unit_to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn u8_to_unit(value: u8) -> f32 {
    f32::from(value) / 255.0
}
