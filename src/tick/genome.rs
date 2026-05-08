use serde::{Deserialize, Serialize};

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PopulationKind {
    Predator = 0,
    Prey = 1,
}
