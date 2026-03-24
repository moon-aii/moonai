#pragma once

#include "core/types.hpp"

#include <vector>
#include <cstdint>

namespace moonai {

class SpatialGrid {
public:
    struct FlatEntry {
        AgentId id;
        float x;
        float y;
    };

    SpatialGrid(int world_width, int world_height, float cell_size);

    void clear();
    void insert(AgentId id, Vec2 position);
    void update(AgentId id, Vec2 position);
    void remove(AgentId id);
    bool contains(AgentId id) const;

    // Returns all agent IDs in the cell containing `position` and its neighbors
    std::vector<AgentId> query(Vec2 position) const;

    // Returns all agent IDs within `radius` of `position`
    std::vector<AgentId> query_radius(Vec2 position, float radius) const;
    void query_radius_into(Vec2 position, float radius, std::vector<AgentId>& result) const;
    void flatten(std::vector<int>& cell_offsets, std::vector<FlatEntry>& entries) const;

    float cell_size() const { return cell_size_; }
    int cols() const { return cols_; }
    int rows() const { return rows_; }
    int world_width() const { return world_width_; }
    int world_height() const { return world_height_; }

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
    std::vector<int> id_to_cell_;
    std::vector<int> id_to_index_;
    std::vector<uint8_t> id_active_;
};

} // namespace moonai
