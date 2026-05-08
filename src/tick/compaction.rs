use serde::{Deserialize, Serialize};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FreeListStateReadback {
    pub tick: u32,
    pub predator_free_slots: u32,
    pub prey_free_slots: u32,
    pub active_food_count: u32,
    pub food_capacity: u32,
}

use crate::tick::genome::PopulationKind;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionSummaryReadback {
    pub population_kind: PopulationKind,
    pub previous_capacity: u32,
    pub live_count: u32,
    pub free_slots_after: u32,
    pub compacted: u32,
}
