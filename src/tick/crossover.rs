use serde::{Deserialize, Serialize};

use crate::tick::genome::PopulationKind;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrossoverSummaryReadback {
    pub population_kind: PopulationKind,
    pub parent_a_slot: u32,
    pub parent_b_slot: u32,
    pub offspring_slot: u32,
    pub offspring_entity_id: u32,
    pub offspring_generation: u32,
    pub inherited_connections: u32,
    pub matching_genes: u32,
    pub disjoint_genes: u32,
    pub excess_genes: u32,
    pub offspring_genome_hash: u64,
}
