#pragma once

#include <cstdint>

namespace moonai {

// uint32_t sizes
namespace sizes {
constexpr float PREDATOR_RADIUS = 2.4f;
constexpr float PREY_RADIUS = 2.0f;
constexpr float FOOD_RADIUS = 1.2f;
} // namespace sizes

// UI Chart colors (SFML RGB)
namespace chart_colors {
constexpr uint8_t PREDATOR_R = 0xFF;
constexpr uint8_t PREDATOR_G = 0x6B;
constexpr uint8_t PREDATOR_B = 0x35;

constexpr uint8_t PREY_R = 0x4E;
constexpr uint8_t PREY_G = 0xCD;
constexpr uint8_t PREY_B = 0xC4;

constexpr uint8_t FOOD_R = 0xaa;
constexpr uint8_t FOOD_G = 0xff;
constexpr uint8_t FOOD_B = 0x6D;
} // namespace chart_colors

// Visualization constants
namespace visual {
// Grid and background
constexpr uint8_t GRID_R = 35;
constexpr uint8_t GRID_G = 32;
constexpr uint8_t GRID_B = 39;

constexpr uint8_t BORDER_R = 78;
constexpr uint8_t BORDER_G = 74;
constexpr uint8_t BORDER_B = 83;

constexpr uint8_t BG_R = 0;
constexpr uint8_t BG_G = 0;
constexpr uint8_t BG_B = 0;

// UI Panel
constexpr uint8_t PANEL_BG_R = 16;
constexpr uint8_t PANEL_BG_G = 13;
constexpr uint8_t PANEL_BG_B = 20;
constexpr uint8_t PANEL_ALPHA = 120;

constexpr uint8_t PANEL_OUTLINE_R = 35;
constexpr uint8_t PANEL_OUTLINE_G = 32;
constexpr uint8_t PANEL_OUTLINE_B = 39;

// Vision range
constexpr uint8_t VISION_FILL_ALPHA = 15;
constexpr uint8_t VISION_OUTLINE_ALPHA = 40;
// Note: VISION_RANGE is configurable via config.lua, not a constant

// Sensor lines
constexpr uint8_t SENSOR_ALPHA = 80;
constexpr uint8_t FOOD_SENSOR_ALPHA = 60;
} // namespace visual

// Renderer constants
namespace visual {
constexpr uint8_t FOOD_ALPHA = 180;
constexpr float SELECTED_OUTLINE_THICKNESS = 2.0f;
constexpr int CIRCLE_POINT_COUNT = 20;
constexpr uint8_t VISION_FILL_R = 255;
constexpr uint8_t VISION_FILL_G = 255;
constexpr uint8_t VISION_FILL_B = 255;
constexpr int VISION_POINT_COUNT = 60;

// Triangle geometry multipliers
constexpr float TRIANGLE_TIP_FACTOR = 1.5f;
constexpr float TRIANGLE_BASE_FACTOR = 0.8f;
constexpr float TRIANGLE_WIDTH_FACTOR = 0.7f;
} // namespace visual

// UI Colors
namespace ui {
// Title and headings
constexpr uint8_t TITLE_R = 200;
constexpr uint8_t TITLE_G = 200;
constexpr uint8_t TITLE_B = 255;

// Fitness/accent
constexpr uint8_t FITNESS_R = 255;
constexpr uint8_t FITNESS_G = 220;
constexpr uint8_t FITNESS_B = 100;

// Muted/secondary text
constexpr uint8_t MUTED_R = 180;
constexpr uint8_t MUTED_G = 180;
constexpr uint8_t MUTED_B = 180;

// Paused indicator
constexpr uint8_t PAUSE_R = 255;
constexpr uint8_t PAUSE_G = 150;
constexpr uint8_t PAUSE_B = 100;

// Bar chart alpha
constexpr uint8_t BAR_ALPHA = 180;

// Event colors
constexpr uint8_t EVENT_KILL_R = 220;
constexpr uint8_t EVENT_KILL_G = 100;
constexpr uint8_t EVENT_KILL_B = 100;

constexpr uint8_t EVENT_FOOD_R = 100;
constexpr uint8_t EVENT_FOOD_G = 220;
constexpr uint8_t EVENT_FOOD_B = 100;

constexpr uint8_t EVENT_BIRTH_R = 100;
constexpr uint8_t EVENT_BIRTH_G = 180;
constexpr uint8_t EVENT_BIRTH_B = 220;

constexpr uint8_t EVENT_DEATH_R = 180;
constexpr uint8_t EVENT_DEATH_G = 180;
constexpr uint8_t EVENT_DEATH_B = 180;

// Fitness chart lines
constexpr uint8_t CHART_BEST_R = 100;
constexpr uint8_t CHART_BEST_G = 150;
constexpr uint8_t CHART_BEST_B = 255;

constexpr uint8_t CHART_AVG_R = 100;
constexpr uint8_t CHART_AVG_G = 220;
constexpr uint8_t CHART_AVG_B = 100;

// NN Visualization
constexpr uint8_t NN_NODE_OUTLINE_R = 200;
constexpr uint8_t NN_NODE_OUTLINE_G = 200;
constexpr uint8_t NN_NODE_OUTLINE_B = 200;
constexpr uint8_t NN_NODE_OUTLINE_A = 120;

constexpr uint8_t NN_INPUT_R = 80;
constexpr uint8_t NN_INPUT_G = 120;
constexpr uint8_t NN_INPUT_B = 220;

constexpr uint8_t NN_BIAS_R = 60;
constexpr uint8_t NN_BIAS_G = 180;
constexpr uint8_t NN_BIAS_B = 220;

constexpr uint8_t NN_HIDDEN_R = 220;
constexpr uint8_t NN_HIDDEN_G = 200;
constexpr uint8_t NN_HIDDEN_B = 80;

constexpr uint8_t NN_OUTPUT_R = 220;
constexpr uint8_t NN_OUTPUT_G = 80;
constexpr uint8_t NN_OUTPUT_B = 80;

// Energy bucket colors (gray gradient)
constexpr uint8_t ENERGY_BUCKET_0_R = 60;
constexpr uint8_t ENERGY_BUCKET_0_G = 60;
constexpr uint8_t ENERGY_BUCKET_0_B = 60;

constexpr uint8_t ENERGY_BUCKET_1_R = 100;
constexpr uint8_t ENERGY_BUCKET_1_G = 100;
constexpr uint8_t ENERGY_BUCKET_1_B = 100;

constexpr uint8_t ENERGY_BUCKET_2_R = 140;
constexpr uint8_t ENERGY_BUCKET_2_G = 140;
constexpr uint8_t ENERGY_BUCKET_2_B = 140;

constexpr uint8_t ENERGY_BUCKET_3_R = 180;
constexpr uint8_t ENERGY_BUCKET_3_G = 180;
constexpr uint8_t ENERGY_BUCKET_3_B = 180;

constexpr uint8_t ENERGY_BUCKET_4_R = 220;
constexpr uint8_t ENERGY_BUCKET_4_G = 220;
constexpr uint8_t ENERGY_BUCKET_4_B = 220;
} // namespace ui

} // namespace moonai
