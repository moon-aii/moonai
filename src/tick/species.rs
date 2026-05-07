use serde::{Deserialize, Serialize};

use crate::tick::genome::PopulationKind;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpeciesSummaryReadback {
    pub population_kind: PopulationKind,
    pub species_id: u32,
    pub size: u32,
    pub representative_slot: u32,
    pub avg_complexity: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepresentativeGenomeHeader {
    pub population_kind: PopulationKind,
    pub slot: u32,
    pub entity_id: u32,
    pub generation: u32,
    pub species_id: u32,
    pub num_nodes: u16,
    pub num_connections: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeciesBatchReadbackHeader {
    pub population_kind: PopulationKind,
    pub species_count: u32,
    pub returned_species_count: u32,
}
