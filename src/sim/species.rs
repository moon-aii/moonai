use serde::{Deserialize, Serialize};

use crate::sim::simulation::PopulationKind;

pub const MAX_SPECIES_SUMMARIES: u32 = 64;

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
pub struct GenomeNodeReadback {
    pub id: u32,
    pub node_type: u8,
    pub reserved0: u8,
    pub reserved1: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GenomeConnectionReadback {
    pub from_node: i32,
    pub to_node: i32,
    pub weight: f32,
    pub innovation: u32,
    pub enabled: u8,
    pub reserved0: u8,
    pub reserved1: u16,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepresentativeGenomeReadback {
    pub header: RepresentativeGenomeHeader,
    pub nodes: Vec<GenomeNodeReadback>,
    pub connections: Vec<GenomeConnectionReadback>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeciesBatchReadbackHeader {
    pub population_kind: PopulationKind,
    pub species_count: u32,
    pub returned_species_count: u32,
}
