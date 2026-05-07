use crate::types::{ConnectionGene, NodeGene};

pub struct Genome {
    pub nodes: Vec<NodeGene>,
    pub connections: Vec<ConnectionGene>,
    pub num_inputs: usize,
    pub num_outputs: usize,
}

impl Genome {
    pub const fn new(num_inputs: usize, num_outputs: usize) -> Self {
        Self { nodes: Vec::new(), connections: Vec::new(), num_inputs, num_outputs }
    }
}
