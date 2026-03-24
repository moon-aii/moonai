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
    std::fill(id_active_.begin(), id_active_.end(), static_cast<uint8_t>(0));
    std::fill(id_to_cell_.begin(), id_to_cell_.end(), -1);
    std::fill(id_to_index_.begin(), id_to_index_.end(), -1);
}

void SpatialGrid::insert(AgentId id, Vec2 position) {
    if (id < id_active_.size() && id_active_[id]) {
        update(id, position);
        return;
    }
    if (id >= id_to_cell_.size()) {
        const size_t new_size = static_cast<size_t>(id) + 1;
        id_to_cell_.resize(new_size, -1);
        id_to_index_.resize(new_size, -1);
        id_active_.resize(new_size, static_cast<uint8_t>(0));
    }
    int idx = cell_index(position.x, position.y);
    if (idx >= 0 && idx < static_cast<int>(cells_.size())) {
        cells_[idx].push_back({id, position});
        id_to_cell_[id] = idx;
        id_to_index_[id] = static_cast<int>(cells_[idx].size()) - 1;
        id_active_[id] = 1;
    }
}

void SpatialGrid::update(AgentId id, Vec2 position) {
    if (id >= id_active_.size() || !id_active_[id]) {
        insert(id, position);
        return;
    }

    const int old_cell = id_to_cell_[id];
    const int new_cell = cell_index(position.x, position.y);
    if (old_cell < 0 || old_cell >= static_cast<int>(cells_.size())
        || new_cell < 0 || new_cell >= static_cast<int>(cells_.size())) {
        return;
    }

    int index_in_cell = id_to_index_[id];
    if (old_cell == new_cell) {
        cells_[old_cell][index_in_cell].pos = position;
        return;
    }

    Entry moved = cells_[old_cell][index_in_cell];
    const int last_index = static_cast<int>(cells_[old_cell].size()) - 1;
    if (index_in_cell != last_index) {
        const Entry swapped = cells_[old_cell][last_index];
        cells_[old_cell][index_in_cell] = swapped;
        id_to_index_[swapped.id] = index_in_cell;
    }
    cells_[old_cell].pop_back();

    moved.pos = position;
    cells_[new_cell].push_back(moved);
    id_to_cell_[id] = new_cell;
    id_to_index_[id] = static_cast<int>(cells_[new_cell].size()) - 1;
}

void SpatialGrid::remove(AgentId id) {
    if (id >= id_active_.size() || !id_active_[id]) {
        return;
    }
    const int cell = id_to_cell_[id];
    const int index_in_cell = id_to_index_[id];
    if (cell < 0 || cell >= static_cast<int>(cells_.size())) {
        return;
    }
    const int last_index = static_cast<int>(cells_[cell].size()) - 1;
    if (index_in_cell != last_index) {
        const Entry swapped = cells_[cell][last_index];
        cells_[cell][index_in_cell] = swapped;
        id_to_index_[swapped.id] = index_in_cell;
    }
    cells_[cell].pop_back();
    id_to_cell_[id] = -1;
    id_to_index_[id] = -1;
    id_active_[id] = 0;
}

bool SpatialGrid::contains(AgentId id) const {
    return id < id_active_.size() && id_active_[id] != 0;
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
    MOONAI_PROFILE_INC(ProfileCounter::GridQueryCalls);
    result.clear();
    float r2 = radius * radius;
    int cells_to_check = static_cast<int>(std::ceil(radius / cell_size_));
    int cx = cell_x(position.x);
    int cy = cell_y(position.y);

    for (int dy = -cells_to_check; dy <= cells_to_check; ++dy) {
        for (int dx = -cells_to_check; dx <= cells_to_check; ++dx) {
            int nx = cx + dx;
            int ny = cy + dy;
            if (nx < 0 || nx >= cols_ || ny < 0 || ny >= rows_) continue;
            int idx = ny * cols_ + nx;
            MOONAI_PROFILE_INC(ProfileCounter::GridCandidatesScanned,
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

void SpatialGrid::flatten(std::vector<int>& cell_offsets, std::vector<SpatialGrid::FlatEntry>& entries) const {
    cell_offsets.clear();
    entries.clear();

    cell_offsets.reserve(cells_.size() + 1);
    cell_offsets.push_back(0);

    std::size_t total_entries = 0;
    for (const auto& cell : cells_) {
        total_entries += cell.size();
    }
    entries.reserve(total_entries);

    for (const auto& cell : cells_) {
        for (const auto& entry : cell) {
            entries.push_back({entry.id, entry.pos.x, entry.pos.y});
        }
        cell_offsets.push_back(static_cast<int>(entries.size()));
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
