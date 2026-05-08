use serde::{Deserialize, Serialize};

use crate::tick::genome::PopulationKind;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReproductionSummaryReadback {
    pub population_kind: PopulationKind,
    pub eligible_parents: u32,
    pub candidate_pairs: u32,
    pub births: u32,
    pub failed_pairs: u32,
}
