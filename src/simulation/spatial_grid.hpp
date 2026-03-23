#pragma once

#include "core/types.hpp"

#include <vector>
#include <cstdint>

namespace moonai {

class SpatialGrid {
public:
    SpatialGrid(int world_width, int world_height, float cell_size);

    void clear();
    void insert(AgentId id, Vec2 position);

    // Returns all agent IDs in the cell containing `position` and its neighbors
    std::vector<AgentId> query(Vec2 position) const;

    // Returns all agent IDs within `radius` of `position`
    std::vector<AgentId> query_radius(Vec2 position, float radius) const;
    void query_radius_into(Vec2 position, float radius, std::vector<AgentId>& result) const;

private:
    int cell_index(float x, float y) const;
    int cell_x(float x) const;
    int cell_y(float y) const;

    struct Entry { AgentId id; Vec2 pos; };

    float cell_size_;
    int cols_;
    int rows_;
    int world_width_;
    int world_height_;
    std::vector<std::vector<Entry>> cells_;
};

} // namespace moonai
