#include "simulation/spatial_grid.hpp"

#include "core/profiler.hpp"

#include <algorithm>
#include <cmath>

namespace moonai {

SpatialGrid::SpatialGrid(int world_width, int world_height, float cell_size)
    : cell_size_(cell_size)
    , world_width_(world_width)
    , world_height_(world_height) {
    cols_ = static_cast<int>(std::ceil(world_width / cell_size_));
    rows_ = static_cast<int>(std::ceil(world_height / cell_size_));
    cells_.resize(cols_ * rows_);
}

void SpatialGrid::clear() {
    for (auto& cell : cells_) {
        cell.clear();
    }
}

void SpatialGrid::insert(AgentId id, Vec2 position) {
    int idx = cell_index(position.x, position.y);
    if (idx >= 0 && idx < static_cast<int>(cells_.size())) {
        cells_[idx].push_back({id, position});
    }
}

std::vector<AgentId> SpatialGrid::query(Vec2 position) const {
    std::vector<AgentId> result;
    int cx = cell_x(position.x);
    int cy = cell_y(position.y);

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = cx + dx;
            int ny = cy + dy;
            if (nx < 0 || nx >= cols_ || ny < 0 || ny >= rows_) continue;
            int idx = ny * cols_ + nx;
            for (const auto& entry : cells_[idx]) {
                result.push_back(entry.id);
            }
        }
    }

    return result;
}

std::vector<AgentId> SpatialGrid::query_radius(Vec2 position, float radius) const {
    std::vector<AgentId> result;
    query_radius_into(position, radius, result);
    return result;
}

void SpatialGrid::query_radius_into(Vec2 position, float radius, std::vector<AgentId>& result) const {
    ScopedTimer timer(ProfileEvent::SpatialQueryRadius);
    Profiler::instance().increment(ProfileCounter::GridQueryCalls);
    result.clear();
    float r2 = radius * radius;
    int cells_to_check = static_cast<int>(std::ceil(radius / cell_size_)) + 1;
    int cx = cell_x(position.x);
    int cy = cell_y(position.y);

    for (int dy = -cells_to_check; dy <= cells_to_check; ++dy) {
        for (int dx = -cells_to_check; dx <= cells_to_check; ++dx) {
            int nx = cx + dx;
            int ny = cy + dy;
            if (nx < 0 || nx >= cols_ || ny < 0 || ny >= rows_) continue;
            int idx = ny * cols_ + nx;
            Profiler::instance().increment(ProfileCounter::GridCandidatesScanned,
                                           static_cast<std::int64_t>(cells_[idx].size()));
            for (const auto& entry : cells_[idx]) {
                float ddx = entry.pos.x - position.x;
                float ddy = entry.pos.y - position.y;
                if (ddx * ddx + ddy * ddy <= r2) {
                    result.push_back(entry.id);
                }
            }
        }
    }
}

int SpatialGrid::cell_index(float x, float y) const {
    return cell_y(y) * cols_ + cell_x(x);
}

int SpatialGrid::cell_x(float x) const {
    int cx = static_cast<int>(x / cell_size_);
    return std::clamp(cx, 0, cols_ - 1);
}

int SpatialGrid::cell_y(float y) const {
    int cy = static_cast<int>(y / cell_size_);
    return std::clamp(cy, 0, rows_ - 1);
}

} // namespace moonai
