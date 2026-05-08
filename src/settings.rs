use crate::config::ConfigError;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub fn load_settings(root_dir: &Path) -> Result<UiConfig, ConfigError> {
    let settings_path = root_dir.join("config").join("settings.json");
    let file = match std::fs::File::open(&settings_path) {
        Ok(f) => f,
        Err(_) => return Ok(UiConfig::default()),
    };
    let settings: UiConfig = serde_json::from_reader(file)?;
    Ok(settings)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiConfig {
    #[serde(default = "default_predator_radius")]
    pub predator_radius: f32,
    #[serde(default = "default_prey_radius")]
    pub prey_radius: f32,
    #[serde(default = "default_food_radius")]
    pub food_radius: f32,
    #[serde(default = "default_predator_color")]
    pub predator_color: [f32; 3],
    #[serde(default = "default_prey_color")]
    pub prey_color: [f32; 3],
    #[serde(default = "default_food_color")]
    pub food_color: [f32; 3],
    #[serde(default = "default_grid_color")]
    pub grid_color: [f32; 3],
    #[serde(default = "default_border_color")]
    pub border_color: [f32; 3],
    #[serde(default = "default_bg_color")]
    pub bg_color: [f32; 3],
    #[serde(default = "default_panel_bg_color")]
    pub panel_bg_color: [f32; 3],
    #[serde(default = "default_panel_alpha")]
    pub panel_alpha: f32,
    #[serde(default = "default_panel_outline_color")]
    pub panel_outline_color: [f32; 3],
    #[serde(default = "default_vision_fill_alpha")]
    pub vision_fill_alpha: f32,
    #[serde(default = "default_vision_outline_alpha")]
    pub vision_outline_alpha: f32,
    #[serde(default = "default_vision_fill_color")]
    pub vision_fill_color: [f32; 4],
    #[serde(default = "default_sensor_alpha")]
    pub sensor_alpha: f32,
    #[serde(default = "default_food_sensor_alpha")]
    pub food_sensor_alpha: f32,
    #[serde(default = "default_food_alpha")]
    pub food_alpha: f32,
    #[serde(default = "default_selected_outline_thickness")]
    pub selected_outline_thickness: f32,
    #[serde(default = "default_circle_point_count")]
    pub circle_point_count: u32,
    #[serde(default = "default_vision_point_count")]
    pub vision_point_count: u32,
    #[serde(default = "default_triangle_tip_factor")]
    pub triangle_tip_factor: f32,
    #[serde(default = "default_triangle_base_factor")]
    pub triangle_base_factor: f32,
    #[serde(default = "default_triangle_width_factor")]
    pub triangle_width_factor: f32,
    #[serde(default = "default_title_color")]
    pub title_color: [f32; 3],
    #[serde(default = "default_fitness_color")]
    pub fitness_color: [f32; 3],
    #[serde(default = "default_muted_color")]
    pub muted_color: [f32; 3],
    #[serde(default = "default_pause_color")]
    pub pause_color: [f32; 3],
    #[serde(default = "default_bar_alpha")]
    pub bar_alpha: f32,
    #[serde(default = "default_event_kill_color")]
    pub event_kill_color: [f32; 3],
    #[serde(default = "default_event_food_color")]
    pub event_food_color: [f32; 3],
    #[serde(default = "default_event_birth_color")]
    pub event_birth_color: [f32; 3],
    #[serde(default = "default_event_death_color")]
    pub event_death_color: [f32; 3],
    #[serde(default = "default_chart_best_color")]
    pub chart_best_color: [f32; 3],
    #[serde(default = "default_chart_avg_color")]
    pub chart_avg_color: [f32; 3],
    #[serde(default = "default_nn_node_outline_color")]
    pub nn_node_outline_color: [f32; 4],
    #[serde(default = "default_nn_input_color")]
    pub nn_input_color: [f32; 3],
    #[serde(default = "default_nn_bias_color")]
    pub nn_bias_color: [f32; 3],
    #[serde(default = "default_nn_hidden_color")]
    pub nn_hidden_color: [f32; 3],
    #[serde(default = "default_nn_output_color")]
    pub nn_output_color: [f32; 3],
    #[serde(default = "default_energy_bucket_0")]
    pub energy_bucket_0: [f32; 3],
    #[serde(default = "default_energy_bucket_1")]
    pub energy_bucket_1: [f32; 3],
    #[serde(default = "default_energy_bucket_2")]
    pub energy_bucket_2: [f32; 3],
    #[serde(default = "default_energy_bucket_3")]
    pub energy_bucket_3: [f32; 3],
    #[serde(default = "default_energy_bucket_4")]
    pub energy_bucket_4: [f32; 3],
    #[serde(default = "default_ui_side_margin")]
    pub ui_side_margin: f32,
    #[serde(default = "default_simulation_margin")]
    pub simulation_margin: f32,
    #[serde(default = "default_fps_limit")]
    pub fps_limit: u32,
    #[serde(default = "default_window_width")]
    pub window_width: u32,
    #[serde(default = "default_window_height")]
    pub window_height: u32,
    #[serde(default = "default_nn_panel_margin")]
    pub nn_panel_margin: f32,
    #[serde(default = "default_nn_panel_gap")]
    pub nn_panel_gap: f32,
    #[serde(default = "default_selected_info_panel_height")]
    pub selected_info_panel_height: f32,
    #[serde(default = "default_nn_panel_top_reserve")]
    pub nn_panel_top_reserve: f32,
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
            predator_radius: default_predator_radius(),
            prey_radius: default_prey_radius(),
            food_radius: default_food_radius(),
            predator_color: default_predator_color(),
            prey_color: default_prey_color(),
            food_color: default_food_color(),
            grid_color: default_grid_color(),
            border_color: default_border_color(),
            bg_color: default_bg_color(),
            panel_bg_color: default_panel_bg_color(),
            panel_alpha: default_panel_alpha(),
            panel_outline_color: default_panel_outline_color(),
            vision_fill_alpha: default_vision_fill_alpha(),
            vision_outline_alpha: default_vision_outline_alpha(),
            vision_fill_color: default_vision_fill_color(),
            sensor_alpha: default_sensor_alpha(),
            food_sensor_alpha: default_food_sensor_alpha(),
            food_alpha: default_food_alpha(),
            selected_outline_thickness: default_selected_outline_thickness(),
            circle_point_count: default_circle_point_count(),
            vision_point_count: default_vision_point_count(),
            triangle_tip_factor: default_triangle_tip_factor(),
            triangle_base_factor: default_triangle_base_factor(),
            triangle_width_factor: default_triangle_width_factor(),
            title_color: default_title_color(),
            fitness_color: default_fitness_color(),
            muted_color: default_muted_color(),
            pause_color: default_pause_color(),
            bar_alpha: default_bar_alpha(),
            event_kill_color: default_event_kill_color(),
            event_food_color: default_event_food_color(),
            event_birth_color: default_event_birth_color(),
            event_death_color: default_event_death_color(),
            chart_best_color: default_chart_best_color(),
            chart_avg_color: default_chart_avg_color(),
            nn_node_outline_color: default_nn_node_outline_color(),
            nn_input_color: default_nn_input_color(),
            nn_bias_color: default_nn_bias_color(),
            nn_hidden_color: default_nn_hidden_color(),
            nn_output_color: default_nn_output_color(),
            energy_bucket_0: default_energy_bucket_0(),
            energy_bucket_1: default_energy_bucket_1(),
            energy_bucket_2: default_energy_bucket_2(),
            energy_bucket_3: default_energy_bucket_3(),
            energy_bucket_4: default_energy_bucket_4(),
            ui_side_margin: default_ui_side_margin(),
            simulation_margin: default_simulation_margin(),
            fps_limit: default_fps_limit(),
            window_width: default_window_width(),
            window_height: default_window_height(),
            nn_panel_margin: default_nn_panel_margin(),
            nn_panel_gap: default_nn_panel_gap(),
            selected_info_panel_height: default_selected_info_panel_height(),
            nn_panel_top_reserve: default_nn_panel_top_reserve(),
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

const fn default_predator_radius() -> f32 {
    1.2
}
const fn default_prey_radius() -> f32 {
    1.0
}
const fn default_food_radius() -> f32 {
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
const fn default_vision_fill_alpha() -> f32 {
    15.0
}
const fn default_vision_outline_alpha() -> f32 {
    40.0
}
const fn default_vision_fill_color() -> [f32; 4] {
    [1.0, 1.0, 1.0, 1.0]
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
const fn default_circle_point_count() -> u32 {
    20
}
const fn default_vision_point_count() -> u32 {
    60
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
const fn default_title_color() -> [f32; 3] {
    [0.784, 0.784, 1.0]
}
const fn default_fitness_color() -> [f32; 3] {
    [1.0, 0.863, 0.392]
}
const fn default_muted_color() -> [f32; 3] {
    [0.706, 0.706, 0.706]
}
const fn default_pause_color() -> [f32; 3] {
    [1.0, 0.588, 0.392]
}
const fn default_bar_alpha() -> f32 {
    180.0
}
const fn default_event_kill_color() -> [f32; 3] {
    [0.863, 0.392, 0.392]
}
const fn default_event_food_color() -> [f32; 3] {
    [0.392, 0.863, 0.392]
}
const fn default_event_birth_color() -> [f32; 3] {
    [0.392, 0.706, 0.863]
}
const fn default_event_death_color() -> [f32; 3] {
    [0.706, 0.706, 0.706]
}
const fn default_chart_best_color() -> [f32; 3] {
    [0.392, 0.588, 1.0]
}
const fn default_chart_avg_color() -> [f32; 3] {
    [0.392, 0.863, 0.392]
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
const fn default_energy_bucket_0() -> [f32; 3] {
    [0.235, 0.235, 0.235]
}
const fn default_energy_bucket_1() -> [f32; 3] {
    [0.392, 0.392, 0.392]
}
const fn default_energy_bucket_2() -> [f32; 3] {
    [0.549, 0.549, 0.549]
}
const fn default_energy_bucket_3() -> [f32; 3] {
    [0.784, 0.784, 0.784]
}
const fn default_energy_bucket_4() -> [f32; 3] {
    [0.863, 0.863, 0.863]
}
const fn default_ui_side_margin() -> f32 {
    300.0
}
const fn default_simulation_margin() -> f32 {
    25.0
}
const fn default_fps_limit() -> u32 {
    120
}
const fn default_window_width() -> u32 {
    1920
}
const fn default_window_height() -> u32 {
    1080
}
const fn default_nn_panel_margin() -> f32 {
    25.0
}
const fn default_nn_panel_gap() -> f32 {
    10.0
}
const fn default_selected_info_panel_height() -> f32 {
    100.0
}
const fn default_nn_panel_top_reserve() -> f32 {
    145.0
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
