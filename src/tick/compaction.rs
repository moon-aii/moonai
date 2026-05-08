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
